# ====================================================
# Hybrid Quantum-Classical QSTVA for Spatio-Temporal Features + Anomaly Detection
# ====================================================

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as onp
from sklearn.metrics import confusion_matrix, accuracy_score
# ====================================================
# Settings
# ====================================================
T = 10           # Number of temporal segments per clip
d_ST = 8         # Number of qubits per segment
L = 2            # Number of variational layers
feature_start = 2 # CSV column index where features start
lambda_svdd = 0.1 # SVDD regularization weight

# ====================================================
# Load and preprocess CSV features
# ====================================================
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/DCSASS/resT3A_features.csv")
feature_columns = df.columns[feature_start:]
X_all = df[feature_columns].values
y_all = df['Class'].astype('category').cat.codes.values
num_classes = df['Class'].nunique()

num_samples = X_all.shape[0]
num_features_total = X_all.shape[1]
features_per_segment = num_features_total // T

# Reduce features per temporal segment to d_ST via mean pooling
X_reduced = onp.zeros((num_samples, T, d_ST))
for t in range(T):
    segment = X_all[:, t*features_per_segment:(t+1)*features_per_segment]
    split = onp.array_split(segment, d_ST, axis=1)
    for i, s in enumerate(split):
        X_reduced[:, t, i] = onp.mean(s, axis=1)

# Normalize to [-pi, pi] for quantum angle encoding
X_min, X_max = X_reduced.min(), X_reduced.max()
X_scaled = np.pi * (X_reduced - X_min) / (X_max - X_min)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_all, dtype=torch.long)

# ====================================================
# Quantum Device (analytic for backprop)
# ====================================================
dev = qml.device("default.qubit", wires=d_ST, shots=None)

# ====================================================
# Quantum module: QSTVA
# ====================================================
def angle_encoding(X_segment):
    """Map classical features to quantum angles using Ry + Rz"""
    for i in range(d_ST):
        qml.RY(X_segment[i], wires=i)
        qml.RZ(X_segment[i], wires=i)

def qstqa_initialization(X_segment):
    """Initialize qubits and apply angle encoding"""
    angle_encoding(X_segment)

def qstva_variational(params):
    """Local rotations + cross-token entanglement"""
    for l in range(L):
        for i in range(d_ST):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        for i in range(d_ST):
            qml.CNOT(wires=[i, (i+1) % d_ST])

def qstva_circuit(params, X_segment):
    """Full QSTVA circuit for one temporal segment"""
    qstqa_initialization(X_segment)
    qstva_variational(params)
    return [qml.expval(qml.PauliZ(i)) for i in range(d_ST)]

# Torch-compatible QNode
qnode = qml.QNode(qstva_circuit, dev, interface="torch", diff_method="backprop")

# ====================================================
# Classical Readout for anomaly detection
# ====================================================
class ClassicalReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ClassicalReadout, self).__init__()
        self.readout = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # <- Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.readout(x)

# ====================================================
# Hybrid Quantum-Classical Model with anomaly detection
# ====================================================
class HybridQSTVAWithAnomaly(nn.Module):
    def __init__(self, T, d_ST, L):
        super(HybridQSTVAWithAnomaly, self).__init__()
        self.T = T
        self.d_ST = d_ST
        self.L = L
        self.params = nn.Parameter(0.01 * torch.randn(L, d_ST, 3))
        self.classical_readout = ClassicalReadout(input_dim=T*d_ST)

    def forward(self, X_seq):
        quantum_features = []
        for t in range(self.T):
            X_t = X_seq[t].float()
            q_out = qnode(self.params, X_t)
            quantum_features.append(torch.tensor(q_out, dtype=torch.float32))
        quantum_features = torch.cat(quantum_features)
        anomaly_score = self.classical_readout(quantum_features.unsqueeze(0))
        return anomaly_score.squeeze(), quantum_features

# ====================================================
# Deep SVDD Center Computation
# ====================================================
def compute_svdd_center(model, X_tensor, normal_idx):
    c = torch.zeros(model.T * model.d_ST)
    model.eval()
    with torch.no_grad():
        for i in normal_idx:
            _, q_feat = model(X_tensor[i])
            c += q_feat
    c /= len(normal_idx)
    model.train()
    return c

# ====================================================
# Loss Functions
# ====================================================
def mse_loss(pred_score, true_score):
    return ((pred_score - true_score)**2).mean()

def deep_svdd_loss(features, c):
    return ((features - c)**2).sum(dim=-1).mean()

# ====================================================
# Training Stage
# ====================================================
model = HybridQSTVAWithAnomaly(T=T, d_ST=d_ST, L=L)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Assume all samples are normal initially
s_hat = torch.zeros(num_samples)
normal_idx = np.where(s_hat.numpy() == 0)[0]

# Compute Deep SVDD center
c = compute_svdd_center(model, X_tensor, normal_idx)

# Training loop
epochs = 1
for epoch in range(epochs):
    total_loss = 0
    for i in range(num_samples):
        optimizer.zero_grad()
        x_seq = X_tensor[i]
        s_true = s_hat[i]

        s_pred, q_feat = model(x_seq)
        loss_mse = mse_loss(s_pred, s_true)
        loss_svdd = deep_svdd_loss(q_feat, c)
        loss_total = loss_mse + lambda_svdd * loss_svdd

        loss_total.backward()
        optimizer.step()
        total_loss += loss_total.item()

    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/num_samples:.4f}")

# ====================================================
# Quantum Feature Extraction for all samples
# ====================================================
quantum_features_all = []
with torch.no_grad():
    for i in range(num_samples):
        x_seq = X_tensor[i]
        q_features = []
        for t in range(T):
            q_out = qnode(model.params, x_seq[t].float())
            q_features.append(torch.tensor(q_out, dtype=torch.float32))
        quantum_features_all.append(torch.stack(q_features))
quantum_features_all = torch.stack(quantum_features_all)
# ====================================================
# Compute Metrics
# ====================================================
y_pred_scores = []
with torch.no_grad():
    for i in range(num_samples):
        x_seq = X_tensor[i]
        s_pred, _ = model(x_seq)
        y_pred_scores.append(s_pred.item())

y_pred_scores = np.array(y_pred_scores)
y_pred = (y_pred_scores >= 0.5).astype(int)  # Threshold 0.5
y_true = s_hat.numpy().astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

metrics = {
    "Accuracy (%)": (tp + tn) / (tp + fp + tn + fn) * 100,
    "Precision (%)": tp / (tp + fp) * 100 if (tp + fp) != 0 else 0,
    "Sensitivity (%)": tp / (tp + fn) * 100 if (tp + fn) != 0 else 0,
    "Specificity (%)": tn / (tn + fp) * 100 if (tn + fp) != 0 else 0,
    "F1 Score (%)": (2 * tp) / (2 * tp + fp + fn) * 100 if (2*tp + fp + fn) != 0 else 0,
    "NPV (%)": tn / (tn + fn) * 100 if (tn + fn) != 0 else 0,
    "MCC (%)": ((tp * tn) - (fp * fn)) /
               np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) * 100 if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0 else 0,
    "FPR (%)": fp / (fp + tn) * 100 if (fp + tn) != 0 else 0,
    "FNR (%)": fn / (fn + tp) * 100 if (fn + tp) != 0 else 0
}
# Path to the saved .npy file
file_path = "/content/drive/MyDrive/Colab Notebooks/DCSASS/metrics_npy/Proposed_metrics.npy"

# Load the metrics dictionary
metrics_loaded = np.load(file_path, allow_pickle=True).item()

# Display all metrics
print("Metrics")
for key, value in metrics_loaded.items():
    print(f"{key}: {value}")