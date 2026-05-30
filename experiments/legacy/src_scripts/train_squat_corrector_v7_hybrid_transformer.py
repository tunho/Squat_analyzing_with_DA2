"""
train_squat_corrector_v7_hybrid_transformer.py

SOTA(최첨단) 모델 (CNN-Transformer Hybrid):
 1. Local-Global Interaction: CNN으로 지역적 특징을, Transformer로 전체적 문맥을 파악
 2. Self-Attention Mechanism: 스쿼트 하강/상승의 중요 지점에 가중치를 자동 부여
 3. Multi-head Attention: 신체 각도와 3D 좌표의 상관관계를 다각도로 해석
 4. Residual Skip-connection: 딥러닝의 정보 손실을 최소화

목표: 3~5도 오차 절대 사수 (세계 최고 수준 성능)
"""
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction/v7_hybrid_transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm"
]

# 1. Dataset (CNN과 동일 구조)
class SquatSequenceDataset(Dataset):
    def __init__(self, data, window_size=10):
        self.windows = []
        self.targets = []
        self.raw_angles = []
        data['raw_knee_angle'] = data['mp_knee_angle'].copy()

        for _, group in data.groupby(['subject_id', 'view_type']):
            group = group.sort_values('frame_index')
            features = group[FEATURE_COLS].values
            residuals = (group['gt_angle'] - group['mp_knee_angle']).values
            raw_angles = group['raw_knee_angle'].values

            for i in range(len(group) - window_size + 1):
                self.windows.append(features[i:i+window_size])
                self.targets.append(residuals[i+window_size-1])
                self.raw_angles.append(raw_angles[i+window_size-1])

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.raw_angles = np.array(self.raw_angles, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), torch.tensor([self.targets[idx]]), self.raw_angles[idx]

# 2. Transformer-CNN Hybrid Model
class SquatTransformer(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2):
        super(SquatTransformer, self).__init__()
        # CNN Feature Extractor
        self.cnn = nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [Batch, Window, Features] -> [Batch, Features, Window] for CNN
        x = x.transpose(1, 2)
        x = torch.relu(self.cnn(x))
        # -> [Batch, d_model, Window] -> [Batch, Window, d_model] for Transformer
        x = x.transpose(1, 2)
        x = self.transformer(x)
        # 윈도우의 마지막 시점 특징만 사용
        x = x[:, -1, :]
        return self.fc(x)

def run_loso_v7(df):
    subjects = sorted(df["subject_id"].unique())
    loso_results = []
    
    # Global Scaling
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    print("\n" + "="*80)
    print("🚀 v7 HYBRID SOTA (CNN + Transformer) LOSO 시작")
    print("="*80)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj]

        train_ds = SquatSequenceDataset(train_df, WINDOW_SIZE)
        test_ds = SquatSequenceDataset(test_df, WINDOW_SIZE)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        model = SquatTransformer(len(FEATURE_COLS))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # 10 Epoch 학습
        model.train()
        for epoch in range(10):
            for batch_x, batch_y, _ in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        all_p, all_g, all_r = [], [], []
        with torch.no_grad():
            for bx, by, br in test_loader:
                pred_res = model(bx).squeeze().numpy()
                all_p.extend(br.numpy() + pred_res)
                all_g.extend(br.numpy() + by.squeeze().numpy())
                all_r.extend(br.numpy())

        mae_raw = mean_absolute_error(all_g, all_r)
        mae_corr = mean_absolute_error(all_g, all_p)
        print(f"  [{test_subj}] Raw MAE: {mae_raw:.2f}° → ⚡ SOTA Corrected: {mae_corr:.2f}°")
        loso_results.append({"subject": test_subj, "mae_raw": mae_raw, "mae_corr": mae_corr})

    res_df = pd.DataFrame(loso_results)
    print("\n" + "-"*80)
    print(f"📊 v7 SOTA 평균 MAE: {res_df['mae_corr'].mean():.2f}°")
    print("="*80)
    return res_df

def main():
    if not INPUT_CSV.exists(): return
    df = pd.read_csv(INPUT_CSV)
    loso = run_loso_v7(df)
    loso.to_csv(OUTPUT_DIR / "loso_transformer_v7.csv", index=False)
    print(f"\n✅ SOTA 하이브리드 고도화 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
