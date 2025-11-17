# ====================================================
# Q3D Algorithm: Quantum 3D Feature Extraction
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
# Quantum Device
# ====================================================
dev = qml.device("default.qubit", wires=d_ST, shots=None)

# ====================================================
# Q3D Quantum Circuit
# ====================================================
def angle_encoding(X_segment):
    """Map classical features to quantum angles using Ry + Rz"""
    for i in range(d_ST):
        qml.RY(X_segment[i], wires=i)
        qml.RZ(X_segment[i], wires=i)

def q3d_initialization(X_segment):
    """Initialize qubits and encode features"""
    angle_encoding(X_segment)

def q3d_variational(params):
    """Variational layers + entanglement"""
    for l in range(L):
        for i in range(d_ST):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        # Circular entanglement
        for i in range(d_ST):
            qml.CNOT(wires=[i, (i+1)%d_ST])

def q3d_circuit(params, X_segment):
    """Full Q3D circuit for a temporal segment"""
    q3d_initialization(X_segment)
    q3d_variational(params)
    return [qml.expval(qml.PauliZ(i)) for i in range(d_ST)]

# Torch-compatible QNode
qnode = qml.QNode(q3d_circuit, dev, interface="torch", diff_method="backprop")

# ====================================================
# Classical Readout
# ====================================================
class ClassicalReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.readout(x)

# ====================================================
# Hybrid Quantum-Classical Q3D Model
# ====================================================
class HybridQ3DModel(nn.Module):
    def __init__(self, T, d_ST, L):
        super().__init__()
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
# Training
# ====================================================
model = HybridQ3DModel(T=T, d_ST=d_ST, L=L)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Placeholder: all normal
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

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/num_samples:.4f}")

# ====================================================
# Extract Quantum Features for All Samples
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
print("Quantum features shape:", quantum_features_all.shape)
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
file_path = "/content/drive/MyDrive/Colab Notebooks/DCSASS/metrics_npy/Q3D_18_metrics.npy"

# Load the metrics dictionary
metrics_loaded = np.load(file_path, allow_pickle=True).item()

# Display all metrics
print("Metrics")
for key, value in metrics_loaded.items():
    print(f"{key}: {value}")

# ====================================================
# J.QCNN Algorithm: Quantum Convolutional Neural Network with Metrics
# ====================================================

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as onp
from sklearn.metrics import confusion_matrix

# ====================================================
# Settings
# ====================================================
T = 10           # Number of temporal segments per clip
d_ST = 8         # Number of qubits per segment
L = 2            # Number of variational layers per quantum conv
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

# Temporal segment mean pooling
X_reduced = onp.zeros((num_samples, T, d_ST))
for t in range(T):
    segment = X_all[:, t*features_per_segment:(t+1)*features_per_segment]
    split = onp.array_split(segment, d_ST, axis=1)
    for i, s in enumerate(split):
        X_reduced[:, t, i] = onp.mean(s, axis=1)

# Normalize to [-pi, pi]
X_min, X_max = X_reduced.min(), X_reduced.max()
X_scaled = np.pi * (X_reduced - X_min) / (X_max - X_min)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_all, dtype=torch.long)

# ====================================================
# Quantum Device
# ====================================================
dev = qml.device("default.qubit", wires=d_ST, shots=None)

# ====================================================
# J.QCNN Quantum Convolutional Circuit
# ====================================================
def angle_encoding(X_segment):
    """Encode classical features into qubits using Ry + Rz"""
    for i in range(d_ST):
        qml.RY(X_segment[i], wires=i)
        qml.RZ(X_segment[i], wires=i)

def quantum_conv_layer(params, wires):
    """Single quantum convolution layer"""
    for i, w in enumerate(wires):
        qml.RX(params[i,0], wires=w)
        qml.RY(params[i,1], wires=w)
        qml.RZ(params[i,2], wires=w)
    # Entanglement between neighboring qubits
    for i in range(len(wires)-1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

def jqcnn_circuit(params, X_segment):
    """Full J.QCNN circuit for one temporal segment"""
    angle_encoding(X_segment)
    for l in range(L):
        quantum_conv_layer(params[l], wires=list(range(d_ST)))
    return [qml.expval(qml.PauliZ(i)) for i in range(d_ST)]

# Torch-compatible QNode
qnode = qml.QNode(jqcnn_circuit, dev, interface="torch", diff_method="backprop")

# ====================================================
# Classical Readout
# ====================================================
class ClassicalReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.readout(x)

# ====================================================
# Hybrid J.QCNN Model
# ====================================================
class HybridJQCNNModel(nn.Module):
    def __init__(self, T, d_ST, L):
        super().__init__()
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
# Training
# ====================================================
model = HybridJQCNNModel(T=T, d_ST=d_ST, L=L)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Placeholder: all normal
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

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/num_samples:.4f}")

# ====================================================
# Extract Quantum Features
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
print("Quantum features shape:", quantum_features_all.shape)

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
file_path = "/content/drive/MyDrive/Colab Notebooks/DCSASS/metrics_npy/J.QCNN_19_metrics.npy"

# Load the metrics dictionary
metrics_loaded = np.load(file_path, allow_pickle=True).item()

# Display all metrics
print("Metrics")
for key, value in metrics_loaded.items():
    print(f"{key}: {value}")
# ====================================================
# CRBA (Chronological Random Bat Algorithm)
# ====================================================

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as onp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import math
import random
import time

# ====================================================
# Settings
# ====================================================
T = 10
d_ST = 8
L = 2
feature_start = 2
lambda_svdd = 0.1

# CRBA hyperparams
BAT_POP = 30
MAX_ITER = 200
FREQ_MIN = 0.0
FREQ_MAX = 2.0
ALPHA = 0.9
GAMMA = 0.9
CRONO_LAMBDA = 0.3

seed = 42
random.seed(seed)
onp.random.seed(seed)
np.random.seed(seed)

# ====================================================
# Load CSV features
# ====================================================
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/DCSASS/resT3A_features.csv")
feature_columns = df.columns[feature_start:]
X_all = df[feature_columns].values
y_all = df['Class'].astype('category').cat.codes.values
num_classes = df['Class'].nunique()

num_samples = X_all.shape[0]
num_features_total = X_all.shape[1]
features_per_segment = num_features_total // T

# Temporal segment pooling
X_reduced = onp.zeros((num_samples, T, d_ST))
for t in range(T):
    segment = X_all[:, t*features_per_segment:(t+1)*features_per_segment]
    split = onp.array_split(segment, d_ST, axis=1)
    for i, s in enumerate(split):
        X_reduced[:, t, i] = onp.mean(s, axis=1)

# Normalize [-pi, pi]
X_min, X_max = X_reduced.min(), X_reduced.max()
X_scaled = np.pi * (X_reduced - X_min) / (X_max - X_min)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ====================================================
# Quantum Device
# ====================================================
dev = qml.device("default.qubit", wires=d_ST, shots=None)

def angle_encoding(X_segment):
    for i in range(d_ST):
        qml.RY(X_segment[i], wires=i)
        qml.RZ(X_segment[i], wires=i)

def quantum_conv_layer(params, wires):
    for i, w in enumerate(wires):
        qml.RX(params[i,0], wires=w)
        qml.RY(params[i,1], wires=w)
        qml.RZ(params[i,2], wires=w)
    for i in range(len(wires)-1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

def jqcnn_circuit(params, X_segment):
    angle_encoding(X_segment)
    for l in range(L):
        quantum_conv_layer(params[l], wires=list(range(d_ST)))
    return [qml.expval(qml.PauliZ(i)) for i in range(d_ST)]

qnode = qml.QNode(jqcnn_circuit, dev, interface="torch", diff_method="backprop")

# Quantum params (fixed)
quantum_params_init = 0.01 * torch.randn(L, d_ST, 3, dtype=torch.float32)

# ====================================================
# Extract Quantum Features
# ====================================================
quantum_features_all = []
with torch.no_grad():
    for i in range(num_samples):
        x_seq = X_tensor[i]
        q_features = []
        for t in range(T):
            q_out = qnode(quantum_params_init, x_seq[t].float())
            q_features.append(torch.tensor(q_out, dtype=torch.float32))
        quantum_features_all.append(torch.stack(q_features))

quantum_features_all = torch.stack(quantum_features_all)
print("Quantum features shape:", quantum_features_all.shape)

# Flatten
X_flat = quantum_features_all.reshape(num_samples, T * d_ST).numpy()
y_true = y_all.copy()

# ====================================================
# Chronological Weighting
# ====================================================
segment_weights = onp.linspace(1 - CRONO_LAMBDA, 1 + CRONO_LAMBDA, T)
feature_weights = onp.repeat(segment_weights, d_ST)
X_flat_crono = X_flat * feature_weights

# ====================================================
# CRBA Optimization
# ====================================================
def sigmoid(x):
    return 1.0 / (1.0 + onp.exp(-x))

def bce_loss(y_true, y_pred, eps=1e-12):
    y_pred = onp.clip(y_pred, eps, 1 - eps)
    return -onp.mean(y_true * onp.log(y_pred) + (1 - y_true) * onp.log(1 - y_pred))

D = T * d_ST
dim = D + 1

bats = onp.random.uniform(-1, 1, size=(BAT_POP, dim))
vel = onp.zeros((BAT_POP, dim))
freq = onp.zeros(BAT_POP)
loudness = onp.ones(BAT_POP) * 0.9
pulse_rate = onp.ones(BAT_POP) * 0.1

def fitness_of(bat_vector):
    w = bat_vector[:D]
    b = bat_vector[D]
    logits = onp.dot(X_flat_crono, w) + b
    probs = sigmoid(logits)
    return bce_loss(y_true, probs)

fitness_vals = onp.array([fitness_of(bats[i]) for i in range(BAT_POP)])
best_idx = int(onp.argmin(fitness_vals))
best_bat = bats[best_idx].copy()
best_fitness = fitness_vals[best_idx].copy()

start_time = time.time()
for t_iter in range(MAX_ITER):
    for i in range(BAT_POP):
        freq[i] = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * random.random()
        vel[i] = vel[i] + (bats[i] - best_bat) * freq[i]
        candidate = bats[i] + vel[i]

        if random.random() > pulse_rate[i]:
            eps_local = 0.001 * onp.random.randn(dim)
            candidate = best_bat + eps_local * onp.mean(loudness)

        cand_fitness = fitness_of(candidate)

        if (cand_fitness <= fitness_vals[i]) and (random.random() < loudness[i]):
            bats[i] = candidate
            fitness_vals[i] = cand_fitness
            loudness[i] = ALPHA * loudness[i]
            pulse_rate[i] = pulse_rate[i] * (1 - onp.exp(-GAMMA * t_iter))

        if cand_fitness < best_fitness:
            best_bat = candidate.copy()
            best_fitness = cand_fitness

    if (t_iter + 1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"CRBA iter {t_iter+1}/{MAX_ITER} best BCE: {best_fitness:.6f}")

print("CRBA finished. Best BCE:", best_fitness)

# ====================================================
# Apply Best Linear Readout
# ====================================================
best_w = best_bat[:D]
best_b = best_bat[D]

logits = onp.dot(X_flat_crono, best_w) + best_b
probs = sigmoid(logits)
y_pred = (probs >= 0.5).astype(int)

# ====================================================
# Metrics (Binary or Multi-class Auto)
# ====================================================
unique_classes = onp.unique(y_true)
num_classes = len(unique_classes)

if num_classes == 2:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "Accuracy (%)": (tp + tn) / (tp + fp + tn + fn) * 100,
        "Precision (%)": tp / (tp + fp) * 100 if (tp + fp) != 0 else 0,
        "Sensitivity (%)": tp / (tp + fn) * 100 if (tp + fn) != 0 else 0,
        "Specificity (%)": tn / (tn + fp) * 100 if (tn + fp) != 0 else 0,
        "F1 Score (%)": (2 * tp) / (2 * tp + fp + fn) * 100 if (2*tp + fp + fn) != 0 else 0,
        "NPV (%)": tn / (tn + fn) * 100 if (tn + fn) != 0 else 0,
        "MCC (%)": ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) * 100,
        "FPR (%)": fp / (fp + tn) * 100,
        "FNR (%)": fn / (fn + tp) * 100
    }

else:
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
        "Precision_macro (%)": precision_score(y_true, y_pred, average='macro') * 100,
        "Recall_macro (%)": recall_score(y_true, y_pred, average='macro') * 100,
        "F1_macro (%)": f1_score(y_true, y_pred, average='macro') * 100,
        "Precision_micro (%)": precision_score(y_true, y_pred, average='micro') * 100,
        "Recall_micro (%)": recall_score(y_true, y_pred, average='micro') * 100,
        "F1_micro (%)": f1_score(y_true, y_pred, average='micro') * 100,
        "MCC (%)": matthews_corrcoef(y_true, y_pred) * 100,
        "Confusion_Matrix": cm
    }
# Path to the saved .npy file
file_path = "/content/drive/MyDrive/Colab Notebooks/DCSASS/metrics_npy/CRBA_20_metrics.npy"

# Load the metrics dictionary
metrics_loaded = np.load(file_path, allow_pickle=True).item()

# Display all metrics
print("Metrics")
for key, value in metrics_loaded.items():
    print(f"{key}: {value}")
# ====================================================
# Algorithm-3D ConvNet Readout (Fixed for Multi-Class)
# ====================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

# -------------------------
# Chronological Weighting
# -------------------------
segment_weights = onp.linspace(1 - CRONO_LAMBDA, 1 + CRONO_LAMBDA, T)
segment_weights_torch = torch.tensor(segment_weights, dtype=torch.float32)
segment_weights_expanded = segment_weights_torch.unsqueeze(1).repeat(1, d_ST)

X_weighted = quantum_features_all * segment_weights_expanded  # (N,T,d_ST)
X_conv = X_weighted.unsqueeze(1).unsqueeze(-1).float()        # (N,1,T,d_ST,1)


# ====================================================
# 3D ConvNet Model
# ====================================================
class Algorithm3DConvNet(nn.Module):
    def __init__(self, T, d_ST, num_classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.Conv3d(8, 16, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=(2,2,1)),

            nn.Conv3d(16, 32, kernel_size=(3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        test_input = torch.zeros(1, 1, T, d_ST, 1)
        out = self.model(test_input)
        flat_dim = out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)     # Multi-class
        )

    def forward(self, x):
        feat = self.model(x)
        feat_flat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat_flat)
        return logits


# ====================================================
# Prepare labels (multi-class)
# ====================================================
y_true = y_all.copy()   # 0..(num_classes-1)
y_tensor = torch.tensor(y_true, dtype=torch.long)   # Correct dtype for CE Loss

# ====================================================
# Train 3D ConvNet Readout
# ====================================================
model3D = Algorithm3DConvNet(T=T, d_ST=d_ST, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model3D.parameters(), lr=0.001)

epochs = 15
batch_size = 16

dataset = torch.utils.data.TensorDataset(X_conv, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model3D(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss / len(loader):.4f}")


# ====================================================
# Evaluate
# ====================================================
with torch.no_grad():
    logits = model3D(X_conv)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

metrics = {
    "Accuracy (%)": (tp + tn) / (tp + fp + tn + fn) * 100,
    "Precision (%)": tp / (tp + fp) * 100 if (tp + fp) != 0 else 0,
    "Sensitivity (%)": tp / (tp + fn) * 100 if (tp + fn) != 0 else 0,
    "Specificity (%)": tn / (tn + fp) * 100 if (tn + fp) != 0 else 0,
    "F1 Score (%)": (2 * tp) / (2 * tp + fp + fn) * 100 if (2*tp + fp + fn) != 0 else 0,
}
# Path to the saved .npy file
file_path = "/content/drive/MyDrive/Colab Notebooks/DCSASS/metrics_npy/3D_ConvNet_23_metrics.npy"

# Load the metrics dictionary
metrics_loaded = np.load(file_path, allow_pickle=True).item()

# Display all metrics
print("Metrics")
for key, value in metrics_loaded.items():
    print(f"{key}: {value}")