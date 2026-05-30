"""
generate_presentation_plots.py

교수님 보고용 최종 3종 세트 그래프를 생성합니다:
  1. v1 vs GT 곡선 (기하학 보정의 효과)
  2. v5 vs GT 곡선 (AI 보정의 핵심 성과)
  3. MAE & RMSE 통합 바 차트 ([Raw, v1, v5] 3단계 성능 비교)
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
    # 1. 데이터 로드 및 통계 계산
    geom_df = pd.read_csv(GEOM_CSV, encoding='utf-8-sig')
    rich_df = pd.read_csv(RICH_CSV)
    df_v5 = prepare_v5_data(rich_df.copy())
    
    with open(MODEL_V5, "rb") as f:
        bundle = pickle.load(f)
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    
    # v5 보정 수행 (전체 데이터 대상)
    X_s = scaler.transform(df_v5[features])
    df_v5['corrected_angle'] = df_v5['mp_knee_angle'] + model.predict(X_s)
    
    # [지표 산출]
    # Raw - v5와 동일한 샘플을 대상으로 계산하여 공정성 유지
    raw_mae = mean_absolute_error(df_v5['gt_angle'], df_v5['mp_knee_angle'])
    raw_rmse = np.sqrt(mean_squared_error(df_v5['gt_angle'], df_v5['mp_knee_angle']))
    
    # v1 (Geometric)
    v1_mae = geom_df.abs_error.mean()
    v1_rmse = np.sqrt(geom_df.sq_error.mean())
    
    # v5 (Proposed)
    v5_mae = mean_absolute_error(df_v5['gt_angle'], df_v5['corrected_angle'])
    v5_rmse = np.sqrt(mean_squared_error(df_v5['gt_angle'], df_v5['corrected_angle']))
    
    # ==============================================================
    # 그래프 1: v1 vs GT (PM_043 샘플)
    # ==============================================================
    plot_v1 = geom_df.iloc[1000:1400].reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.plot(plot_v1['frame_index'], plot_v1['gt_angle'], 'k--', label='Ground Truth', alpha=0.5)
    plt.plot(plot_v1['frame_index'], plot_v1['pred_angle'], '#4CAF50', label='Proposed v1: Geometric Repair', linewidth=2.5)
    plt.title("Method 1: Geometric Rule-based (v1) Performance Chart", fontsize=14)
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(RESULTS_DIR / "presentation_plot_01_v1.png", dpi=200)
    plt.close()

    # ==============================================================
    # 그래프 2: v5 vs GT (PM_043 샘플)
    # ==============================================================
    subset_v5 = df_v5[(df_v5['subject_id'] == 'PM_043') & (df_v5['view_type'] == 'side')].sort_values('frame_index')
    plot_v5 = subset_v5.iloc[1000:1400].reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.plot(plot_v5['frame_index'], plot_v5['gt_angle'], 'k--', label='Ground Truth', alpha=0.5)
    plt.plot(plot_v5['frame_index'], plot_v5['corrected_angle'], '#2196F3', label='Proposed v5: Biomechanical AI', linewidth=3)
    plt.title("Method 2: Final Biomechanical AI (v5) Performance Chart", fontsize=14)
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(RESULTS_DIR / "presentation_plot_02_v5.png", dpi=200)
    plt.close()

    # ==============================================================
    # 그래프 3: [MAE, RMSE] 통합 바 차트
    # ==============================================================
    labels = ['Baseline (Raw)', 'Method 1 (v1)', 'Method 2 (v5/Proposed)']
    maes = [raw_mae, v1_mae, v5_mae]
    rmses = [raw_rmse, v1_rmse, v5_rmse]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, maes, width, label='MAE (Mean Absolute Error)', color='#9E9E9E', alpha=0.8)
    rects2 = ax.bar(x + width/2, rmses, width, label='RMSE (Root Mean Squared Error)', color='#FF5722', alpha=0.8)
    
    ax.set_ylabel('Degrees (deg)', fontsize=12)
    ax.set_title('Step-by-Step Performance Evolution', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    
    # 숫자 표기
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}°', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    autolabel(rects1)
    autolabel(rects2)
    
    plt.ylim(0, 20)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "presentation_plot_03_bars.png", dpi=200)
    plt.close()
    
    print(f"✅ 교수님 보고용 3종 세트 그래프 생성 완료! 위치: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
