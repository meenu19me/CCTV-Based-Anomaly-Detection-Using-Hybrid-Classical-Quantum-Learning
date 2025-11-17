import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# 1. Res-T3A Network Definition
# -------------------------------
class SpatialResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights="IMAGENET1K_V2")
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 2048

    def forward(self, x):  # x: [B,T,3,H,W]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        feat = self.feature_extractor(x)  # [B*T, 2048, 1, 1]
        feat = feat.view(B, T, self.out_dim)  # [B, T, 2048]
        return feat, feat  # Return feat twice for visualization

class TCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.conv(x)
        return self.relu(y)

class TCN(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        self.block1 = TCNBlock(input_dim, hidden_dim, dilation=1)
        self.block2 = TCNBlock(hidden_dim, hidden_dim, dilation=2)
        self.out_dim = hidden_dim
    def forward(self, F_s):
        x = F_s.permute(0,2,1)
        x = self.block1(x)
        x = self.block2(x)
        return x.permute(0,2,1)

class I3D(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.out_dim = out_dim
    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        feat = self.conv3d(x)
        feat = feat.mean(dim=[3,4])
        return feat.permute(0,2,1)

class TemporalAttention(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.out_dim = input_dim
    def forward(self, F_s):
        score = self.v(torch.tanh(self.W(F_s))).squeeze(-1)
        alpha = F.softmax(score, dim=1).unsqueeze(-1)
        context = (alpha * F_s).sum(dim=1)
        context = context.unsqueeze(1).repeat(1, F_s.size(1), 1)
        return context

class ResT3A(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialResNet50()
        self.tcn = TCN()
        self.i3d = I3D()
        self.att = TemporalAttention()
        self.proj_TCN = nn.Linear(self.tcn.out_dim, 256)
        self.proj_I3D = nn.Linear(self.i3d.out_dim, 256)
        self.proj_ATT = nn.Linear(2048, 256)
        self.out_dim = 256*3
    def forward(self, x):
        F_s, spatial_feat_map = self.spatial(x)  # Get spatial features
        F_tcn = self.tcn(F_s)
        F_i3d = self.i3d(x)
        F_att = self.att(F_s)
        F_tcn = self.proj_TCN(F_tcn)
        F_i3d = self.proj_I3D(F_i3d)
        F_att = self.proj_ATT(F_att)
        X_ST = torch.cat([F_tcn, F_i3d, F_att], dim=-1)
        return X_ST, spatial_feat_map  # return for visualization

# -------------------------------
# Load Images
# -------------------------------
def load_images_from_main_folder(main_folder, extensions=('.png','.jpg','.jpeg','.bmp','.tiff')):
    images_info = []
    images_data = []
    for root, dirs, files in os.walk(main_folder):
        class_name = os.path.basename(root)
        for file in sorted(files):
            if file.lower().endswith(extensions):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images_info.append([class_name, file, img_path])
                    images_data.append((img, class_name, file, img_path))
    return images_info, images_data

# -------------------------------
# Paths
# -------------------------------
main_folder = "/content/drive/MyDrive/Colab Notebooks/DCSASS/Augmented_Display"
feature_csv_path = "/content/drive/MyDrive/Colab Notebooks/DCSASS/resT3A_features.csv"
os.makedirs(os.path.dirname(feature_csv_path), exist_ok=True)

images_info, images_data = load_images_from_main_folder(main_folder)

# -------------------------------
# Initialize Model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResT3A().to(device)
model.eval()

# -------------------------------
# Extract Features & Display Layer Maps
# -------------------------------
all_features = []

for img, class_name, filename, path in tqdm(images_data):
    # Convert to tensor [1,1,3,H,W]
    img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        features, spatial_feat_map = model(img_tensor)  # [B,T,768], [B,T,2048]

    # Average over T frames (here T=1)
    features_np = features.squeeze(0).mean(axis=0).cpu().numpy()  # [768]
    all_features.append([class_name, filename] + features_np.tolist())

    # -------------------------------
    # Display Input Image
    # -------------------------------
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Input: {filename}")
    plt.axis("off")
    plt.show()

    # -------------------------------
    # Display some spatial feature maps (first 16 channels)
    # -------------------------------
    spatial_map = spatial_feat_map.squeeze(0).cpu().numpy()  # [T,2048]
    plt.figure(figsize=(12,6))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(spatial_map[0,i].reshape(1,1), cmap='viridis')  # visualize as small heatmap
        plt.axis("off")
        plt.title(f"Ch-{i}")
    plt.suptitle(f"Spatial Feature Maps: {filename}")
    plt.show()

# -------------------------------
# Save CSV
# -------------------------------
columns = ["Class","Filename"] + [f"F_{i}" for i in range(all_features[0][2:].__len__())]
df_features = pd.DataFrame(all_features, columns=columns)
df_features.to_csv(feature_csv_path, index=False)
print("Feature CSV saved at:", feature_csv_path)
display(df_features.head())
df_features