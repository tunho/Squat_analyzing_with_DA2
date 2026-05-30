"""
generate_pm008_full_sequence_plots.py

PM_008 피험자의 전체 영상 구간(0~5200 프레임)에 대해:
  1. 기하학 보정(v1) 성과 대조 그래프 (v1 vs GT)
  2. 최종 AI 보정(v5) 성과 대조 그래프 (v5 vs GT)
  3. 사용자 요청 레이아웃(검은색 GT, 파란색 보정값) 반영
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

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
    TARGET_SUBJ = 'PM_008'
    
    # 1. 1세대 기하학(v1) 로드
    geom_df = pd.read_csv(GEOM_CSV, encoding='utf-8-sig') # PM_008 분석용
    
    # 2. 최종 AI(v5) 로드 및 보정
    rich_df = pd.read_csv(RICH_CSV)
    df_v5 = prepare_v5_data(rich_df.copy())
    df_subj = df_v5[(df_v5['subject_id'] == TARGET_SUBJ) & (df_v5['view_type'] == 'side')].copy()
    
    with open(MODEL_V5, "rb") as f:
        bundle = pickle.load(f)
    model, scaler, ft = bundle["model"], bundle["scaler"], bundle["features"]
    X_s = scaler.transform(df_subj[ft])
    df_subj['corrected_angle'] = df_subj['mp_knee_angle'] + model.predict(X_s)

    # -------------------------------------------------------------
    # 그래프 1: v1 (기하학) vs GT (전체 구간)
    # -------------------------------------------------------------
    plt.figure(figsize=(20, 8))
    plt.plot(geom_df['frame_index'], geom_df['gt_angle'], 'k-', label='GT (ref)', alpha=0.9, linewidth=1.2)
    plt.plot(geom_df['frame_index'], geom_df['pred_angle'], '#6495ED', label='v1 Geometric Repair', alpha=0.9, linewidth=1.0)
    
    plt.title(f"Knee Angle - {TARGET_SUBJ}_full_frame_results (Method 1: Geometric Repair)", fontsize=16)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Angle (°)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.ylim(0, 200)
    plt.xlim(0, 5500)
    plt.legend(loc='upper right', frameon=True)
    
    out1 = RESULTS_DIR / f"pm008_v1_full_sequence.png"
    plt.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 그래프 2: v5 (AI) vs GT (전체 구간)
    # -------------------------------------------------------------
    plt.figure(figsize=(20, 8))
    # AI 보정 시 프레임 손실(Shift/Rolling) 반영하여 원본과 동기화
    plt.plot(df_subj['frame_index'], df_subj['gt_angle'], 'k-', label='GT (ref)', alpha=0.9, linewidth=1.2)
    plt.plot(df_subj['frame_index'], df_subj['corrected_angle'], '#4169E1', label='v5 Biomechanical AI', alpha=1.0, linewidth=1.2)
    
    plt.title(f"Knee Angle - {TARGET_SUBJ}_full_frame_results (Method 2: Proposed AI v5)", fontsize=16)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Angle (°)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.ylim(0, 200)
    plt.xlim(0, 5500)
    plt.legend(loc='upper right', frameon=True)
    
    out2 = RESULTS_DIR / f"pm008_v5_full_sequence.png"
    plt.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ PM_008 전 구간(5200 frames) 비교 그래프 2종 생성 완료!")

if __name__ == "__main__":
    main()
