"""
train_squat_corrector_v3_advanced.py

고도화 기법 적용 (기존 3개 피처 데이터 활용):
 1. Residual Learning (잔차 학습): 실제 각도가 아닌 MediaPipe의 '오차'를 예측
 2. Lag Features (시계열 데이터): 이전 프레임의 각도를 피처로 사용 (Smoothing 효과)
 3. View-Specific Modeling: 정면/측면 여부를 주요 피처로 명시적 활용
 4. Ensemble: GradientBoosting + RandomForest Stacking

목표: MAE를 3~5도 수준으로 최대한 근접하게 개선
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction/v3_advanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 데이터 로드 및 전처리
def prepare_data(df):
    # 정렬 (시계열 피처 생성을 위해 중요)
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    
    # 시계열 피처 (Lag Features) - 이전 1~2프레임의 각도 차이
    df['mp_angle_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_angle'].shift(1)
    df['mp_angle_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_angle'].shift(2)
    
    # 널값 처리 (첫 프레임들)
    df = df.dropna(subset=['mp_angle_lag1', 'mp_angle_lag2'])
    
    # View Type 수치화
    df['view_idx'] = df['view_type'].map({'front': 0, 'side': 1})
    
    # 타겟 설정: Residual (잔차) = GT - MP
    df['residual'] = df['gt_angle'] - df['mp_angle']
    
    return df

# 학습에 사용할 피처 리스트
FEATURE_COLS = [
    "mp_angle", "hip_depth_norm", "hip_ankle_dist_norm", 
    "view_idx", "mp_angle_lag1", "mp_angle_lag2"
]
TARGET_COL = "residual"

def run_loso_v3(df):
    subjects = sorted(df["subject_id"].unique())
    results = []

    print("\n" + "="*70)
    print("🚀 고급 기법(Residual + Lag + Multi-Model) LOSO 교차검증 시작")
    print("="*70)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj].copy()
        test_df = df[df["subject_id"] == test_subj].copy()

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df["gt_angle"] # 평가는 실제 각도로

        # 스케일링
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 앙상블 모델 (GBM + RF)
        gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        gbm.fit(X_train_s, y_train)
        rf.fit(X_train_s, y_train)

        # 예측 가중 합 (Stacking 효과)
        pred_res_gbm = gbm.predict(X_test_s)
        pred_res_rf = rf.predict(X_test_s)
        pred_res = (0.7 * pred_res_gbm) + (0.3 * pred_res_rf)

        # 모델 보정: MP_Angle + Predicted_Residual
        y_corrected = test_df["mp_angle"].to_numpy() + pred_res
        
        mae_raw = mean_absolute_error(y_test, test_df["mp_angle"])
        mae_corrected = mean_absolute_error(y_test, y_corrected)
        
        print(f"  [{test_subj}] Raw: {mae_raw:.2f}° → Corrected: {mae_corrected:.2f}° (개선: {mae_raw-mae_corrected:.2f}°)")
        
        results.append({
            "subject": test_subj,
            "mae_raw": mae_raw,
            "mae_corrected": mae_corrected
        })

    res_df = pd.DataFrame(results)
    print("\n" + "-"*70)
    print(f"✨ LOSO 평균 개선: {res_df['mae_raw'].mean():.2f}° → {res_df['mae_corrected'].mean():.2f}°")
    print("="*70)
    return res_df

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} 파일이 없습니다.")
        return

    raw_df = pd.read_csv(INPUT_CSV)
    df = prepare_data(raw_df)
    
    loso_results = run_loso_v3(df)
    loso_results.to_csv(OUTPUT_DIR / "loso_advanced_v3.csv", index=False)
    
    # 전체 데이터로 최종 모델 학습 및 저장
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    final_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    final_model.fit(X_s, y)
    
    bundle = {
        "model": final_model,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "method": "residual_v3"
    }
    with open(OUTPUT_DIR / "correction_model_v3.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"\n✅ 고도화 완료! 모델 저장: {OUTPUT_DIR / 'correction_model_v3.pkl'}")

if __name__ == "__main__":
    main()
