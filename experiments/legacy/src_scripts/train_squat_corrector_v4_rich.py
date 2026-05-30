"""
train_squat_corrector_v4_rich.py

최종 고도화 버젼 (17개 Rich Features 전용):
 1. 전신 기하 정보 활용: 무릎, 발목, 어깨의 3D 좌표 및 가시성(Visibility) 점수 포함
 2. 골반-무릎 복합 보정: Hip Angle과 Knee Angle의 상관관계를 통한 보정
 3. Residual & Lag Learning: 이전 2프레임 히스토리와 오차(Delta) 예측 기반
 4. 강력한 앙상블: ExtraTrees + GradientBoosting + View-Specific Fine-tuning

목표: Q1 논문 수준의 MAE 3~5도 달성 시도
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction/v4_rich_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 학습에 사용할 전체 피처 (17개)
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis",
    "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2" # 시계열 추가 예정
]
TARGET_COL = "residual"

def prepare_rich_data(df):
    # 시계열 정렬
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    
    # Lag Features 추가
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    
    # 잔차 타겟 계산 (GT - MP)
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    
    # View 인코딩
    df['view_idx'] = df['view_type'].map({'front': 0, 'side': 1})
    
    # 결측치 제거
    df = df.dropna(subset=['knee_lag1', 'knee_lag2'])
    return df

def train_loso_v4(df):
    subjects = sorted(df["subject_id"].unique())
    loso_results = []

    print("\n" + "="*75)
    print("💎 최종 고도화 (17 Features + ExtraTrees Ensemble) LOSO 교차검증 시작")
    print("="*75)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj].copy()

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df["gt_angle"]

        # 스케일링
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 모델 1: ExtraTrees (좌표 데이터에 강력함)
        et = ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
        # 모델 2: GradientBoosting (잔차 학습에 강력함)
        gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.07, max_depth=5, random_state=42)

        et.fit(X_train_s, y_train)
        gbm.fit(X_train_s, y_train)

        # 앙상블 (ET 0.6 + GBM 0.4)
        pred_res = (0.6 * et.predict(X_test_s)) + (0.4 * gbm.predict(X_test_s))
        
        # 보정 각도 산출
        y_corrected = test_df["mp_knee_angle"].to_numpy() + pred_res
        
        mae_raw = mean_absolute_error(y_test, test_df["mp_knee_angle"])
        mae_corrected = mean_absolute_error(y_test, y_corrected)
        
        print(f"  [{test_subj}] Raw MAE: {mae_raw:.2f}° → Corrected MAE: {mae_corrected:.2f}° (개선: {mae_raw-mae_corrected:.2f}°)")
        
        loso_results.append({
            "subject": test_subj,
            "mae_raw": mae_raw,
            "mae_corrected": mae_corrected,
            "improvement": mae_raw - mae_corrected
        })

    loso_df = pd.DataFrame(loso_results)
    print("\n" + "-"*75)
    print(f"✨ 최종 LOSO 평균 개선: {loso_df['mae_raw'].mean():.2f}° → {loso_df['mae_corrected'].mean():.2f}°")
    print(f"🔥 최종 목표 도달도: {(loso_df['mae_corrected'].mean() <= 5.0)}")
    print("="*75)
    return loso_df

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} 파일이 없습니다.")
        return

    raw_df = pd.read_csv(INPUT_CSV)
    df = prepare_rich_data(raw_df)
    
    results = train_loso_v4(df)
    results.to_csv(OUTPUT_DIR / "loso_rich_v4.csv", index=False)
    
    # 7:2 Hold-out 최종 모델 빌드 및 저장
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_s, y)
    
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "method": "rich_residual_v4"
    }
    with open(OUTPUT_DIR / "final_correction_model_v4.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"\n✅ 최종 모델 탄생! 저장 경로: {OUTPUT_DIR / 'final_correction_model_v4.pkl'}")

if __name__ == "__main__":
    main()
