"""
train_squat_corrector_v5_paper.py

학술지(Q1) 투고급 고도화 모델 (Biomechanical Phase-Aware Expert System):
 1. Anatomical Normalization: 피험자별 대퇴골/경골 길이 비율(Femur/Tibia Ratio) 피처 도입
 2. Phase-Specific Experts: 하강(Descent), 최하단(Bottom), 상승(Ascent) 구간별 독립 전문가 모델 운영
 3. Velocity-Informed Residual: 관절의 움직임 속도(Velocity)를 반영한 동적 오차 보정
 4. Robust LOSO: 1명 피험자를 완전히 제외한 상태에서의 일반화 성능 극대화

목표: 임상적 골드 스탠다드인 MAE 3~5도 달성
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 생체역학적 정보 통합 및 위상 분류 (Preprocessing)
def prepare_paper_data(df):
    # 시간 순 정렬
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    
    # [1] 신체 비율(Anatomical Ratio) 계산
    # k_x, k_y, k_z (무릎) / a_x, a_y, a_z (발목) 는 이미 엉덩이(h) 기준 정규화 좌표임
    df['femur_len'] = np.sqrt(df['k_x']**2 + df['k_y']**2 + df['k_z']**2)
    df['tibia_len'] = np.sqrt((df['a_x']-df['k_x'])**2 + (df['a_y']-df['k_y'])**2 + (df['a_z']-df['k_z'])**2)
    
    # 피험자별 평균 비율 산출 (고정된 신체적 특징)
    subj_ratio = df.groupby('subject_id')[['femur_len', 'tibia_len']].mean()
    subj_ratio['leg_ratio'] = subj_ratio['femur_len'] / (subj_ratio['tibia_len'] + 1e-6)
    df['leg_ratio'] = df['subject_id'].map(subj_ratio['leg_ratio'])
    
    # [2] 위상(Phase) 분류: 내려가는 중, 올라가는 중, 바닥
    df['angle_velocity'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].diff().fillna(0)
    # 스무딩 처리 (윈도우 5)
    df['angle_vel_smooth'] = df.groupby(['subject_id', 'view_type'])['angle_velocity'].transform(lambda x: x.rolling(5, center=True).mean()).fillna(0)
    
    def classify_phase(vel, angle):
        if angle < 100: return 'bottom' # 무릎이 많이 굽혀진 상태면 bottom으로 우선 분류 (보강)
        if vel < -0.15: return 'descent'
        if vel > 0.15: return 'ascent'
        return 'bottom'
    
    df['phase'] = df.apply(lambda row: classify_phase(row['angle_vel_smooth'], row['mp_knee_angle']), axis=1)
    
    # [3] 타겟: 잔차 (Residual)
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    
    # Lag Features (v4 기법 계승)
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    df = df.dropna(subset=['knee_lag1', 'knee_lag2'])
    
    return df

# 최종 피처 리스트
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle", "leg_ratio", "angle_vel_smooth",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis", "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]

def train_loso_v5(df):
    subjects = sorted(df["subject_id"].unique())
    loso_results = []

    print("\n" + "="*80)
    print("🎓 v5 PAPER-LEVEL (Anatomical Scaling + Phase Experts) LOSO 시작")
    print("="*80)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj].copy()

        # 위상별 전문가 모델 학습 (하강, 하단, 상승)
        experts = {}
        X_test = test_df[FEATURE_COLS]
        y_test_gt = test_df["gt_angle"]
        y_test_mp = test_df["mp_knee_angle"].to_numpy()
        
        # 전체 보정값 저장을 위한 배열
        y_pred_residual = np.zeros(len(test_df))

        for phase in ['descent', 'bottom', 'ascent']:
            phase_train = train_df[train_df['phase'] == phase]
            if phase_train.empty: continue
            
            # 모델: ExtraTrees (강력한 정밀도)
            model = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
            
            # 스케일링
            scaler = StandardScaler()
            X_train_p = scaler.fit_transform(phase_train[FEATURE_COLS])
            y_train_p = phase_train["residual"]
            
            model.fit(X_train_p, y_train_p)
            
            # 테스트 데이터 중 해당 위상인 프레임만 예측
            phase_test_mask = (test_df['phase'] == phase).values
            if phase_test_mask.any():
                X_test_p = scaler.transform(test_df.loc[phase_test_mask, FEATURE_COLS])
                y_pred_residual[phase_test_mask] = model.predict(X_test_p)
        
        # 최종 보정
        y_corrected = y_test_mp + y_pred_residual
        
        mae_raw = mean_absolute_error(y_test_gt, y_test_mp)
        mae_corrected = mean_absolute_error(y_test_gt, y_corrected)
        
        print(f"  [{test_subj}] Raw MAE: {mae_raw:.2f}° → 🔥 v5 Corrected MAE: {mae_corrected:.2f}° (개선: {mae_raw-mae_corrected:.2f}°)")
        
        loso_results.append({
            "subject": test_subj,
            "mae_raw": mae_raw,
            "mae_corrected": mae_corrected
        })

    loso_df = pd.DataFrame(loso_results)
    avg_raw = loso_df['mae_raw'].mean()
    avg_corr = loso_df['mae_corrected'].mean()
    
    print("\n" + "-"*80)
    print(f"📊 최종 LOSO 평균 성과: {avg_raw:.2f}° → {avg_corr:.2f}° (개선율: {(avg_raw-avg_corr)/avg_raw*100:.1f}%)")
    print(f"🌟 논문 투고 적합도: {'VERY HIGH' if avg_corr <= 5.0 else 'GOOD'}")
    print("="*80)
    return loso_df

def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] {INPUT_CSV} 파일이 없습니다.")
        return

    df_rich = pd.read_csv(INPUT_CSV)
    df = prepare_paper_data(df_rich)
    
    results = train_loso_v5(df)
    results.to_csv(OUTPUT_DIR / "loso_paper_v5.csv", index=False)
    
    # 전체 서비스용 통합 모델 (마지막에 전 위상 통합 ExtraTrees로 하나 더 빌드)
    X = df[FEATURE_COLS]
    y = df["residual"]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    final_model = ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    final_model.fit(X_s, y)
    
    bundle = {
        "model": final_model,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "method": "biomechanical_expert_v5"
    }
    with open(OUTPUT_DIR / "paper_correction_model_v5.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"\n✅ 논문급 서비스 모델 완료! 저장 경로: {OUTPUT_DIR / 'paper_correction_model_v5.pkl'}")

if __name__ == "__main__":
    main()
