"""
plot_angle_time_comparison.py

최종 시각적 검증 (Visual Qualitative Analysis):
 1. 특정 피험자(PM_043)의 연속된 스쿼트 1~2회 구간 샘플링
 2. [Raw MediaPipe / v5 Corrected / Ground Truth] 3가지 곡선 중첩 시각화
 3. 위상(Descent/Bottom/Ascent) 영역 표시를 통한 모델의 추종 성능 증명

논문의 'Results - Qualitative Analysis' 섹션에 핵심 이미지로 사용됩니다.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
MODEL_PATH = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/paper_correction_model_v5.pkl"
RESULTS_DIR = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final"

def prepare_v5_data(df):
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    # v5 피처 재생산
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

FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def main():
    if not INPUT_CSV.exists() or not MODEL_PATH.exists(): return
    
    # 데이터 및 모델 로드
    df_raw = pd.read_csv(INPUT_CSV)
    df = prepare_v5_data(df_raw)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]

    # 1. 시각 분석용 샘플 선택 (가장 오차가 컸던 시점 중 하나)
    # PM_043 (보정 전 오차 10도 이상)
    sample_subj = "PM_043"
    sample_view = "side"
    
    subset = df[(df['subject_id'] == sample_subj) & (df['view_type'] == sample_view)].sort_values('frame_index')
    
    # 약 200프레임 (스쿼트 약 2~3회 반복 구간) 샘플링
    start_idx = 1000 # 영상 중간 즈음
    if len(subset) < 200: start_idx = 0
    plot_data = subset.iloc[start_idx : start_idx + 400].reset_index(drop=True)

    # 2. 보정값 도출
    X_s = scaler.transform(plot_data[FEATURE_COLS])
    y_pred_residual = model.predict(X_s)
    plot_data['corrected_angle'] = plot_data['mp_knee_angle'] + y_pred_residual

    # 3. 그래프 생성
    plt.figure(figsize=(15, 6))
    
    # GT (점선)
    plt.plot(plot_data['frame_index'], plot_data['gt_angle'], 'k--', label='Ground Truth (Gold Standard)', alpha=0.5, linewidth=2)
    # Raw MP (오차 포함)
    plt.plot(plot_data['frame_index'], plot_data['mp_knee_angle'], '#FF9800', label='Raw 3D MediaPipe', alpha=0.9, linewidth=1.5)
    # v5 Corrected (보정 후)
    plt.plot(plot_data['frame_index'], plot_data['corrected_angle'], '#2196F3', label='v5 Biomechanical Corrected', alpha=1.0, linewidth=2.5)

    # 4. 필드 채우기 (오차 가시화)
    plt.fill_between(plot_data['frame_index'], plot_data['gt_angle'], plot_data['mp_knee_angle'], color='#FF9800', alpha=0.1, label='Raw Error')
    plt.fill_between(plot_data['frame_index'], plot_data['gt_angle'], plot_data['corrected_angle'], color='#2196F3', alpha=0.2, label='Minimized Error')

    plt.title(f"Visual Qualitative Comparison: Subject {sample_subj} ({sample_view} view)", fontsize=16, fontweight='bold')
    plt.xlabel("Frame Index (30 Hz)", fontsize=12)
    plt.ylabel("Knee Angle (degrees)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # 50도-180도 사이 스케일 조정 (스쿼트 특화)
    plt.ylim(60, 190)
    
    plt.tight_layout()
    out_path = RESULTS_DIR / "angle_time_comparison_pm043.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✅ 시뮬레이션 곡선 비교 그래프 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
