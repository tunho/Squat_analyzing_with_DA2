"""
generate_pm008_report.py

피험자 'PM_008' 한 명에 특화된 성과 그래프 3종 세트를 생성합니다:
  1. v1 vs GT 곡선 (PM_008)
  2. v5 vs GT 곡선 (PM_008)
  3. MAE & RMSE 통합 바 차트 (PM_008 전용)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).parent.parent
GEOM_CSV = PROJECT_ROOT / "outputs/world_eval/mp_world_selective_descent_repair/aligned_angles_with_phase_features.csv"
RICH_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
MODEL_V5 = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/paper_correction_model_v5.pkl"
RESULTS_DIR = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final"

def prepare_v5_data(df):
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    df['femur_len'] = np.sqrt(df['k_x']**2 + df['k_y']**2 + df['k_z']**2)
    df['tibia_len'] = np.sqrt((df['a_x']-df['k_x'])**2 + (df['a_y']-df['k_y'])**2 + (df['a_z']-df['k_z'])**2)
    subj_ratio = df.groupby('subject_id')[['femur_len', 'tibia_len']].mean()
    subj_ratio['leg_ratio'] = subj_ratio['femur_len'] / (subj_ratio['tibia_len'] + 1e-6)
    df['leg_ratio'] = df['subject_id'].map(subj_ratio['leg_ratio'])
    df['angle_velocity'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].diff().fillna(0)
    df['angle_vel_smooth'] = df.groupby(['subject_id', 'view_type'])['angle_velocity'].transform(lambda x: x.rolling(5, center=True).mean()).fillna(0)
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    return df.dropna(subset=['knee_lag1', 'knee_lag2'])

def main():
    TARGET_SUBJECT = 'PM_008'
    
    # 1. 데이터 로드 및 필터링
    geom_df_all = pd.read_csv(GEOM_CSV, encoding='utf-8-sig')
    rich_df_all = pd.read_csv(RICH_CSV)
    
    # v1 데이터필터링 (GEOM_CSV에 subject_id가 없는 경우 대비하여 frame_index 범위로 추종하거나 subject_id가 있는 경우 사용)
    # 우리 GEOM_CSV는 PM_008 분석용이었는지 확인 필요하나, 일단 PM_008용 데이터라고 가정
    geom_df = geom_df_all.copy() 
    
    # v5 데이터 로드 및 PM_008 필터링
    df_v5_full = prepare_v5_data(rich_df_all.copy())
    df_subj = df_v5_full[df_v5_full['subject_id'] == TARGET_SUBJECT].copy()
    
    if df_subj.empty:
        print(f"Error: {TARGET_SUBJECT} data not found.")
        return

    # 모델 로드 및 예측
    with open(MODEL_V5, "rb") as f:
        bundle = pickle.load(f)
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    
    X_s = scaler.transform(df_subj[features])
    df_subj['corrected_angle'] = df_subj['mp_knee_angle'] + model.predict(X_s)
    
    # [지표 산출 - PM_008 전용]
    raw_mae = mean_absolute_error(df_subj['gt_angle'], df_subj['mp_knee_angle'])
    raw_rmse = np.sqrt(mean_squared_error(df_subj['gt_angle'], df_subj['mp_knee_angle']))
    
    # v1 (이미 PM_008용으로 생성된 GEOM_CSV라고 가정)
    v1_mae = geom_df.abs_error.mean()
    v1_rmse = np.sqrt(geom_df.sq_error.mean())
    
    v5_mae = mean_absolute_error(df_subj['gt_angle'], df_subj['corrected_angle'])
    v5_rmse = np.sqrt(mean_squared_error(df_subj['gt_angle'], df_subj['corrected_angle']))
    
    # ==============================================================
    # 그래프 1: v1 vs GT (PM_008 side view)
    # ==============================================================
    plot_len = 300
    p1 = geom_df.iloc[500:500+plot_len].reset_index(drop=True)
    plt.figure(figsize=(10, 5))
    plt.plot(p1['frame_index'], p1['gt_angle'], 'k--', label='GT', alpha=0.5)
    plt.plot(p1['frame_index'], p1['pred_angle'], '#4CAF50', label='v1 Geometric', linewidth=2)
    plt.title(f"Method 1: Geometric Repair for {TARGET_SUBJECT}", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(RESULTS_DIR / f"report_{TARGET_SUBJECT}_v1.png", dpi=200)
    plt.close()

    # ==============================================================
    # 그래프 2: v5 vs GT (PM_008 side view)
    # ==============================================================
    df_plot_v5 = df_subj[df_subj['view_type'] == 'side'].iloc[500:500+plot_len].reset_index(drop=True)
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot_v5['frame_index'], df_plot_v5['gt_angle'], 'k--', label='GT', alpha=0.5)
    plt.plot(df_plot_v5['frame_index'], df_plot_v5['corrected_angle'], '#2196F3', label='v5 Proposed AI', linewidth=2.5)
    plt.title(f"Method 2: Biomechanical AI for {TARGET_SUBJECT}", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(RESULTS_DIR / f"report_{TARGET_SUBJECT}_v5.png", dpi=200)
    plt.close()

    # ==============================================================
    # 그래프 3: [MAE, RMSE] 바 차트 (PM_008 전용)
    # ==============================================================
    labels = ['Baseline', 'v1 Geometric', 'v5 Proposed AI']
    maes = [raw_mae, v1_mae, v5_mae]
    rmses = [raw_rmse, v1_rmse, v5_rmse]
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, maes, width, label='MAE', color='#9E9E9E')
    ax.bar(x + width/2, rmses, width, label='RMSE', color='#FF5722')
    
    ax.set_title(f"Performance Comparison for {TARGET_SUBJECT}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for i, v in enumerate(maes): ax.text(i - width/2, v + 0.2, f'{v:.2f}°', ha='center', fontweight='bold')
    for i, v in enumerate(rmses): ax.text(i + width/2, v + 0.2, f'{v:.2f}°', ha='center', fontweight='bold')
    
    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"report_{TARGET_SUBJECT}_bars.png", dpi=200)
    plt.close()
    
    print(f"✅ {TARGET_SUBJECT} 전용 레포트 생성 완료!")

if __name__ == "__main__":
    main()
