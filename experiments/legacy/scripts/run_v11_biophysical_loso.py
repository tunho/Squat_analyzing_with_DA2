import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
V7_CSV = PROJECT_ROOT / "outputs/ml_correction/v7_universal/v7_loso_results.csv"
V9_CSV = PROJECT_ROOT / "outputs/ml_correction/v9_xgboost/v9_master_comparison.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v11_biophysical"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Feature set
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

def refine_bone_lengths(df):
    """v11 핵심 로직: 뼈 길이 기하학적 제약 조건 적용"""
    print("💎 Skeletal Constraint Refinement 진행 중...")
    
    refined_dfs = []
    # 각 피험자/운동/뷰별로 독립적인 뼈 길이(중앙값) 산출
    groups = df.groupby(['subject_id', 'ex_id', 'view_type'])
    
    for (sid, exid, view), group in groups:
        group = group.copy().sort_values('frame_index')
        
        # 1. 원본 뼈 길이 계산 (이미 힙이 (0,0,0)인 상태)
        k = group[['k_x', 'k_y', 'k_z']].values
        a = group[['a_x', 'a_y', 'a_z']].values
        
        # Thigh = Hip to Knee
        thigh_lens = np.linalg.norm(k, axis=1)
        # Shank = Knee to Ankle
        shank_lens = np.linalg.norm(a - k, axis=1)
        
        # 2. 중앙값(Median) 고정
        m_thigh = np.median(thigh_lens)
        m_shank = np.median(shank_lens)
        
        # 3. 좌표 재투영 (Unit vector * Median length)
        # Knee 보정
        k_refined = (k / thigh_lens[:, None]) * m_thigh
        # Ankle 보정 (무릎으로부터 발목 방향 벡터 유지하며 길이만 고정)
        a_dir = (a - k) / shank_lens[:, None]
        a_refined = k_refined + (a_dir * m_shank)
        
        # 4. 데이터 업데이트
        group[['k_x', 'k_y', 'k_z']] = k_refined
        group[['a_x', 'a_y', 'a_z']] = a_refined
        
        refined_dfs.append(group)
        
    return pd.concat(refined_dfs)

def run_v11_loso(df):
    subjects = sorted(df['subject_id'].unique())
    results = []
    
    print(f"\n🚀 v11 (ExtraTrees + Skeleton Fixed + Smoothing) LOSO 시작...")
    
    for i, test_sid in enumerate(subjects):
        train_df = df[df['subject_id'] != test_sid]
        test_df = df[df['subject_id'] == test_sid].sort_values(['ex_id', 'view_type', 'frame_index'])
        
        if train_df.empty or test_df.empty: continue

        # 모델 학습 (v7 기반 ExtraTrees)
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        
        # 예측
        preds = model.predict(test_df[FEATURES])
        
        # Post-processing: 시간적 스무딩 (각 영상 단위로)
        test_df['pred_angle'] = preds
        refined_preds = []
        
        for (ex_id, view), sub_group in test_df.groupby(['ex_id', 'view_type']):
            p_seq = sub_group['pred_angle'].values
            if len(p_seq) > 11:
                # Savitzky-Golay Filter (Window 11, poly 3)
                p_smooth = savgol_filter(p_seq, 11, 3)
            else:
                p_smooth = p_seq
            
            mae = mean_absolute_error(sub_group['gt_angle'], p_smooth)
            results.append({"subject": test_sid, "ex_id": f"Ex{ex_id}", "view": view, "mae": mae})
            
        print(f"  [{i+1}/{len(subjects)}] Subject {test_sid} 완료. ({(i+1)/len(subjects)*100:.1f}%)", end='\r')
        
    res_df = pd.DataFrame(results)
    summary = res_df.groupby(['ex_id', 'view'])['mae'].mean().reset_index()
    summary.rename(columns={'mae': 'v11_mae'}, inplace=True)
    return summary

def main():
    if not DATA_PATH.exists(): return
    
    # 1. 데이터 로드 및 뼈 길이 정규화
    raw_df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    refined_df = refine_bone_lengths(raw_df)
    
    # 2. LOSO 검증 및 스무딩 적용
    v11_summary = run_v11_loso(refined_df)
    
    # 3. 기준 데이터 로드 (v7, v9)
    def get_mae(csv, col):
        if not os.path.exists(csv): return None
        tmp = pd.read_csv(csv)
        if 'exercise' in tmp.columns: tmp.rename(columns={'exercise':'ex_id'}, inplace=True)
        if 'v9_mae' in tmp.columns: # v9_master form
            return tmp[['ex_id', 'view', col]]
        return tmp.groupby(['ex_id', 'view'])['mae'].mean().reset_index().rename(columns={'mae': col})

    v7_mae = get_mae(V7_CSV, 'v7_mae')
    v9_mae = get_mae(PROJECT_ROOT / "outputs/ml_correction/v9_xgboost/v9_master_comparison.csv", 'v9_mae')

    # 4. 종합 리포트
    report = v11_summary
    if v7_mae is not None: report = pd.merge(report, v7_mae, on=['ex_id', 'view'], how='left')
    if v9_mae is not None: report = pd.merge(report, v9_mae, on=['ex_id', 'view'], how='left')
    
    # 순서 정리
    report = report[['ex_id', 'view', 'v7_mae', 'v9_mae', 'v11_mae']]
    report['Improvement_v7_v11(%)'] = (report['v7_mae'] - report['v11_mae']) / report['v7_mae'] * 100

    print("\n\n--- 🔥 v11 최종 성적표 (Skeletal Refinement + Smoothing) ---")
    print(report.round(3).to_string(index=False))
    
    report.to_csv(SAVE_DIR / "v11_final_master_comparison.csv", index=False)
    print(f"\n✨ v11 최종 성적 저장 완료: {SAVE_DIR}/v11_final_master_comparison.csv")

if __name__ == "__main__":
    main()
