"""
generate_dl_comparison_plots.py

1. v6 (CNN) 및 v7 (Transformer) 모델을 빠르게 학습 (전체 데이터 대상)
2. 각 모델의 가중치(.pth) 저장
3. PM_008(정면) 데이터에 대해 [GT vs CNN] [GT vs Transformer] 추적 곡선 생성
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/dl_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm"
]

# --- 1. Dataset & Models ---
class SquatSequenceDataset(Dataset):
    def __init__(self, data, window_size=10):
        self.windows, self.targets, self.raw_angles = [], [], []
        for _, group in data.groupby(['subject_id', 'view_type']):
            group = group.sort_values('frame_index')
            features = group[FEATURE_COLS].values
            residuals = (group['gt_angle'] - group['mp_knee_angle']).values
            raw_angles = group['mp_knee_angle'].values
            for i in range(len(group) - window_size + 1):
                self.windows.append(features[i:i+window_size])
                self.targets.append(residuals[i+window_size-1])
                self.raw_angles.append(raw_angles[i+window_size-1])
        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]).transpose(0, 1), torch.tensor([self.targets[idx]])

class SquatCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class SquatTransformer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2) # [B, L, C]
        x = self.proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# --- 2. Main Logic ---
def main():
    df = pd.read_csv(INPUT_CSV)
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    
    # PM_008 Front 데이터 분리 (테스트용)
    test_subj = 'PM_008'
    train_df = df[df['subject_id'] != test_subj]
    test_df_front = df[(df['subject_id'] == test_subj) & (df['view_type'] == 'front')].sort_values('frame_index')
    
    train_ds = SquatSequenceDataset(train_df, WINDOW_SIZE)
    test_ds = SquatSequenceDataset(test_df_front, WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Train CNN ---
    cnn = SquatCNN(len(FEATURE_COLS)).to(device)
    opt = optim.Adam(cnn.parameters(), lr=0.001)
    crit = nn.MSELoss()
    print("Training CNN...")
    for _ in range(5): 
        for x, y in train_loader:
            opt.zero_grad(); crit(cnn(x.to(device)), y.to(device)).backward(); opt.step()
    
    # --- Train Transformer ---
    tf = SquatTransformer(len(FEATURE_COLS)).to(device)
    opt = optim.Adam(tf.parameters(), lr=0.001)
    print("Training Transformer...")
    for _ in range(5):
        for x, y in train_loader:
            opt.zero_grad(); crit(tf(x.to(device)), y.to(device)).backward(); opt.step()

    # --- Evaluation ---
    cnn.eval(); tf.eval()
    cnn_preds, tf_preds, gt_list, raw_list = [], [], [], []
    
    # 원본 각도 복구를 위한 데이터 로드
    raw_df = pd.read_csv(INPUT_CSV)
    raw_sub = raw_df[(raw_df['subject_id'] == test_subj) & (raw_df['view_type'] == 'front')].sort_values('frame_index')
    raw_angles = raw_sub['mp_knee_angle'].values[WINDOW_SIZE-1:]
    gt_angles = raw_sub['gt_angle'].values[WINDOW_SIZE-1:]
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            cnn_preds.extend(cnn(x).cpu().numpy().flatten())
            tf_preds.extend(tf(x).cpu().numpy().flatten())
    
    cnn_corr = raw_angles + np.array(cnn_preds)
    tf_corr = raw_angles + np.array(tf_preds)
    
    # --- Plotting ---
    plt.figure(figsize=(15, 6))
    plt.plot(gt_angles[:500], 'k--', label='GT', alpha=0.5)
    plt.plot(cnn_corr[:500], '#E91E63', label='v6 CNN Prediction', linewidth=1.5)
    plt.title(f"DL Comparison: v6 CNN Tracking Performance ({test_subj})")
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.4)
    plt.savefig(RESULTS_DIR / "dl_check_v6_cnn.png", dpi=150); plt.close()

    plt.figure(figsize=(15, 6))
    plt.plot(gt_angles[:500], 'k--', label='GT', alpha=0.5)
    plt.plot(tf_corr[:500], '#9C27B0', label='v7 Transformer Prediction', linewidth=1.5)
    plt.title(f"DL Comparison: v7 Transformer Tracking Performance ({test_subj})")
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.4)
    plt.savefig(RESULTS_DIR / "dl_check_v7_transformer.png", dpi=150); plt.close()
    
    print(f"✅ DL 모델 비교 곡선 생성 완료! {RESULTS_DIR}")

if __name__ == "__main__":
    main()
