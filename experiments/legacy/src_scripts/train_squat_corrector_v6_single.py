import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction/v6_single_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 피처 리스트 (V5와 동일)
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def prepare_v6_data(df):
    """위상 분류를 제거하고 기본 피처만 가공"""
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    
    # 신체 비율(Anatomical Ratio) 계산 (V5 로직 유지)
    df['femur_len'] = np.sqrt(df['k_x']**2 + df['k_y']**2 + df['k_z']**2)
    df['tibia_len'] = np.sqrt((df['a_x']-df['k_x'])**2 + (df['a_y']-df['k_y'])**2 + (df['a_z']-df['k_z'])**2)
    subj_ratio = df.groupby('subject_id')[['femur_len', 'tibia_len']].mean()
    subj_ratio['leg_ratio'] = subj_ratio['femur_len'] / (subj_ratio['tibia_len'] + 1e-6)
    df['leg_ratio'] = df['subject_id'].map(subj_ratio['leg_ratio'])
    
    # 속도 피처
    df['angle_velocity'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df.groupby(['subject_id', 'view_type'])['angle_velocity'].transform(lambda x: x.rolling(5, center=True).mean()).fillna(0)
    
    # Lag Features
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    df = df.dropna(subset=['knee_lag1', 'knee_lag2'])
    
    # 타겟: 잔차 (Residual)
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    
    return df

def train_loso_v6(df):
    subjects = sorted(df["subject_id"].unique())
    loso_results = []

    print("\n🚀 [v6 SINGLE MODEL] (Integrated No-Phase Expert) LOSO 시작")
    print("="*80)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj].copy()

        # V6 핵심: 단일 모델 사용
        X_train = train_df[FEATURE_COLS]
        y_train = train_df["residual"]
        
        X_test = test_df[FEATURE_COLS]
        y_test_gt = test_df["gt_angle"]
        y_test_mp = test_df["mp_knee_angle"].to_numpy()

        # 모델 설정
        model = ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        
        # 보정 수행
        y_pred_residual = model.predict(X_test_scaled)
        y_corrected = y_test_mp + y_pred_residual
        
        # 지표 산출
        mae_raw = mean_absolute_error(y_test_gt, y_test_mp)
        mae_v6 = mean_absolute_error(y_test_gt, y_corrected)
        rmse_raw = np.sqrt(mean_squared_error(y_test_gt, y_test_mp))
        rmse_v6 = np.sqrt(mean_squared_error(y_test_gt, y_corrected))
        
        print(f"  [{test_subj}] Raw MAE: {mae_raw:.2f}° -> v6 MAE: {mae_v6:.2f}° / v6 RMSE: {rmse_v6:.2f}°")
        
        loso_results.append({
            "subject": test_subj,
            "mae_raw": mae_raw,
            "mae_v6": mae_v6,
            "rmse_raw": rmse_raw,
            "rmse_v6": rmse_v6
        })

    loso_df = pd.DataFrame(loso_results)
    avg_mae = loso_df['mae_v6'].mean()
    avg_rmse = loso_df['rmse_v6'].mean()
    
    print("\n" + "-"*50)
    print(f"📊 v6 최종 평균 MAE: {avg_mae:.2f}° / RMSE: {avg_rmse:.2f}°")
    print("="*50)
    return loso_df

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} 파일이 없습니다.")
        return

    df_rich = pd.read_csv(INPUT_CSV)
    df = prepare_v6_data(df_rich)
    
    results = train_loso_v6(df)
    results.to_csv(OUTPUT_DIR / "loso_single_v6.csv", index=False)
    
    # 서비스용 통합 모델
    X = df[FEATURE_COLS]
    y = df["residual"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    final_model = ExtraTreesRegressor(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
    final_model.fit(X_s, y)
    
    bundle = {"model": final_model, "scaler": scaler, "features": FEATURE_COLS, "method": "integrated_single_v6"}
    with open(OUTPUT_DIR / "squat_single_model_v6.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"\n✅ v6 단일 통합 모델 파일 생성 완료: {OUTPUT_DIR / 'squat_single_model_v6.pkl'}")

if __name__ == "__main__":
    main()
