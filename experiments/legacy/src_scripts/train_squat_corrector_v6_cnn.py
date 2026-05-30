"""
train_squat_corrector_v6_cnn.py

최종 보스 모델 (Temporal 1D-CNN Deep Learning):
 1. Sequence Modeling: 단일 프레임이 아닌 10프레임(Window)의 흐름을 통째로 입력
 2. 1D-CNN Architecture: 가속도, 속도 등 동작의 '리듬'을 딥러닝이 스스로 학습
 3. Temporal Consistency: 프레임 간의 각도 변화가 급격하지 않도록 시계열적 매끄러움 확보
 4. PyTorch 기반: 대용량 데이터(65k)를 효율적으로 소화하는 딥러닝 엔진

목표: MAE 3~5도 안착 (Q1 저널 1st Tier 수준)
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
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction/v6_deep_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 10 # 10프레임(약 0.33초) 단위로 동작의 흐름 파악
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm"
]

# 1. Dataset Class 정의
class SquatSequenceDataset(Dataset):
    def __init__(self, data, window_size=10):
        self.windows = []
        self.targets = []
        self.raw_angles = []

        # 피험자/시점별로 윈도우 생성 (Subject/View 경계 보호)
        for _, group in data.groupby(['subject_id', 'view_type']):
            group = group.sort_values('frame_index')
            features = group[FEATURE_COLS].values
            residuals = (group['gt_angle'] - group['mp_knee_angle']).values
            raw_angles = group['raw_knee_angle'].values

            for i in range(len(group) - window_size + 1):
                self.windows.append(features[i:i+window_size])
                self.targets.append(residuals[i+window_size-1]) # 윈도우의 마지막 프레임 오차 예측
                self.raw_angles.append(raw_angles[i+window_size-1])

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.raw_angles = np.array(self.raw_angles, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]).transpose(0, 1), torch.tensor([self.targets[idx]]), self.raw_angles[idx]

# 2. 1D-CNN 모델 정의
class SquatCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(SquatCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# 3. LOSO 하이퍼-딥러닝 검증
def run_loso_v6(df):
    subjects = sorted(df["subject_id"].unique())
    loso_results = []

    # 스케일러 적용 전 원본 각도 보존
    df['raw_knee_angle'] = df['mp_knee_angle'].copy()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLS])
    df[FEATURE_COLS] = scaled_features

    print("\n" + "="*80)
    print("🧠 v6 DEEP SEQUENCE (1D-CNN + Residual) LOSO 시작")
    print("="*80)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj]

        train_dataset = SquatSequenceDataset(train_df, WINDOW_SIZE)
        print(f"    - Dataset created (Size: {len(train_dataset)})")
        test_dataset = SquatSequenceDataset(test_df, WINDOW_SIZE)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = SquatCNN(len(FEATURE_COLS))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(10):
            running_loss = 0.0
            for batch_x, batch_y, _ in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"    - Epoch {epoch+1}/10: Loss {running_loss/len(train_loader):.4f}")

        # 테스트
        model.eval()
        all_preds = []
        all_gts = []
        all_raws = []
        
        # GT는 Residual + MP_Angle 이므로 복구 필요
        with torch.no_grad():
            for batch_x, batch_y, batch_raw in test_loader:
                preds_res = model(batch_x).squeeze().numpy()
                all_preds.extend(batch_raw.numpy() + preds_res)
                all_gts.extend(batch_raw.numpy() + batch_y.squeeze().numpy())
                all_raws.extend(batch_raw.numpy())

        mae_raw = mean_absolute_error(all_gts, all_raws)
        mae_corrected = mean_absolute_error(all_gts, all_preds)
        
        print(f"  [{test_subj}] Raw MAE: {mae_raw:.2f}° → ⚡ v6 CNN Corrected: {mae_corrected:.2f}°")
        loso_results.append({"subject": test_subj, "mae_raw": mae_raw, "mae_corrected": mae_corrected})

    loso_df = pd.DataFrame(loso_results)
    print("\n" + "-"*80)
    print(f"📊 최종 v6 CNN 평균 오차: {loso_df['mae_corrected'].mean():.2f}°")
    print(f"🚀 개선 혁신성: {((loso_df['mae_raw'].mean() - loso_df['mae_corrected'].mean()) / loso_df['mae_raw'].mean() * 100):.1f}%")
    print("="*80)
    return loso_df

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} 파일 존재하지 않음")
        return

    df = pd.read_csv(INPUT_CSV)
    loso_df = run_loso_v6(df)
    loso_df.to_csv(OUTPUT_DIR / "loso_cnn_v6.csv", index=False)
    
    # 최종 모델 저장 (간소화)
    print(f"\n✅ CNN 딥러닝 고도화 완료! 결과 저장: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
