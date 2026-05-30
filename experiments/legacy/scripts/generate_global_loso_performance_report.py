"""
generate_global_loso_performance_report.py

전체 데이터셋(9명 피험자, 약 65,000 프레임)에 대해:
  1. 9-fold LOSO (Leave-One-Subject-Out) 교차 검증 수행
  2. 전체 피험자에 대한 평균 MAE / RMSE 산출
  3. [Raw (Baseline) vs v5 (Proposed LOSO)] 최종 글로벌 성과 막대 그래프 생성
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/global_report"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_v5_features(df):
    df = df.copy()
    df['femur_len'] = np.sqrt(df['k_x']**2 + df['k_y']**2 + df['k_z']**2)
    df['tibia_len'] = np.sqrt((df['a_x']-df['k_x'])**2 + (df['a_y']-df['k_y'])**2 + (df['a_z']-df['k_z'])**2)
    subj_ratio = df.groupby('subject_id')[['femur_len', 'tibia_len']].mean()
    subj_ratio['leg_ratio'] = subj_ratio['femur_len'] / (subj_ratio['tibia_len'] + 1e-6)
    df['leg_ratio'] = df['subject_id'].map(subj_ratio['leg_ratio'])
    df['angle_velocity'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df.groupby(['subject_id', 'view_type'])['angle_velocity'].transform(lambda x: x.rolling(5, center=True).mean()).fillna(0)
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    return df.dropna()

FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def main():
    # 1. 데이터 로드 및 전처리
    print("Loading and preparing global dataset...")
    df_raw = pd.read_csv(INPUT_CSV)
    df = prepare_v5_features(df_raw)
    subjects = sorted(df['subject_id'].unique())
    
    loso_results = []
    
    # 2. 9-fold LOSO 교차 검증 수행
    print(f"Starting 9-fold LOSO Cross-Validation on {len(subjects)} subjects...")
    for test_subj in tqdm(subjects, desc="LOSO Subjects"):
        train_df = df[df['subject_id'] != test_subj]
        test_df = df[df['subject_id'] == test_subj]
        
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURE_COLS])
        y_train = train_df['residual']
        
        # Training
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Prediction (Generalization)
        X_test = scaler.transform(test_df[FEATURE_COLS])
        test_df['pred_residual'] = model.predict(X_test)
        test_df['proposed_angle'] = test_df['mp_knee_angle'] + test_df['pred_residual']
        
        # Metrics per subject
        gt = test_df['gt_angle']
        raw = test_df['mp_knee_angle']
        prop = test_df['proposed_angle']
        
        mae_raw = mean_absolute_error(gt, raw)
        mae_prop = mean_absolute_error(gt, prop)
        rmse_raw = np.sqrt(mean_squared_error(gt, raw))
        rmse_prop = np.sqrt(mean_squared_error(gt, prop))
        
        loso_results.append({
            "subject": test_subj,
            "mae_raw": mae_raw,
            "mae_prop": mae_prop,
            "rmse_raw": rmse_raw,
            "rmse_prop": rmse_prop
        })
        
    results_df = pd.DataFrame(loso_results)
    
    # [글로벌 평균 산출]
    global_mae_raw = results_df.mae_raw.mean()
    global_mae_prop = results_df.mae_prop.mean()
    global_rmse_raw = results_df.rmse_raw.mean()
    global_rmse_prop = results_df.rmse_prop.mean()
    
    # 3. 최종 바 차트 생성 (Baseline vs Global Proposed LOSO)
    labels = ['Baseline (MediaPipe Global)', 'Proposed (Global LOSO Avg)']
    maes = [global_mae_raw, global_mae_prop]
    rmses = [global_rmse_raw, global_rmse_prop]
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 8))
    rects1 = ax.bar(x - width/2, maes, width, label='Overall MAE (Generalization)', color='#BDBDBD', alpha=0.9)
    rects2 = ax.bar(x + width/2, rmses, width, label='Overall RMSE (Robustness)', color='#2196F3', alpha=0.9)
    
    ax.set_title("Global Benchmark: Overall Generalization Performance\n(Average over 9 Subjects, 65k Frames)", fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel("Degrees (°)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.legend(fontsize=12)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')
    autolabel(rects1)
    autolabel(rects2)
    
    plt.ylim(0, 20)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "global_loso_performance_bars.png", dpi=200)
    plt.close()
    
    # 피험자별 비교표 저장
    results_df.to_csv(RESULTS_DIR / "subject_wise_loso_metrics.csv", index=False)
    
    print(f"✅ 최종 글로벌 성과 보고서 생성 완료!")
    print(f"🌍 Overall Baseline MAE: {global_mae_raw:.2f}°")
    print(f"🌍 Overall Proposed(LOSO) MAE: {global_mae_prop:.2f}°")

if __name__ == "__main__":
    main()
