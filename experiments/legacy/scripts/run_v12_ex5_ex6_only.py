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
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v12_ex5_ex6_only"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Features (Same as v11)
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

def refine_bone_lengths(df):
    refined_dfs = []
    groups = df.groupby(['subject_id', 'ex_id', 'view_type'])
    for (_, _, _), group in groups:
        group = group.copy().sort_values('frame_index')
        k = group[['k_x', 'k_y', 'k_z']].values
        a = group[['a_x', 'a_y', 'a_z']].values
        thigh_lens = np.linalg.norm(k, axis=1)
        shank_lens = np.linalg.norm(a - k, axis=1)
        m_thigh = np.median(thigh_lens)
        m_shank = np.median(shank_lens)
        k_refined = (k / (thigh_lens[:, None] + 1e-8)) * m_thigh
        a_dir = (a - k) / (shank_lens[:, None] + 1e-8)
        a_refined = k_refined + (a_dir * m_shank)
        group[['k_x', 'k_y', 'k_z']] = k_refined
        group[['a_x', 'a_y', 'a_z']] = a_refined
        refined_dfs.append(group)
    return pd.concat(refined_dfs)

def run_loso_comparison(df, model_type="v7"):
    subjects = sorted(df['subject_id'].unique())
    results = []
    
    for i, test_sid in enumerate(subjects):
        train_df = df[df['subject_id'] != test_sid]
        test_df = df[df['subject_id'] == test_sid].sort_values(['ex_id', 'view_type', 'frame_index'])
        if train_df.empty or test_df.empty: continue

        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        preds = model.predict(test_df[FEATURES])
        
        test_df['pred_angle'] = preds
        for (ex_id, view), sub_group in test_df.groupby(['ex_id', 'view_type']):
            p_seq = sub_group['pred_angle'].values
            if model_type == "v11":
                # Smoothing applied only for v11 style
                if len(p_seq) > 11: p_seq = savgol_filter(p_seq, 11, 3)
            
            mae = mean_absolute_error(sub_group['gt_angle'], p_seq)
            results.append({"subject": test_sid, "ex_id": f"Ex{ex_id}", "view": view, "mae": mae})
            
        print(f"  [{model_type}] {i+1}/{len(subjects)} Subjects validated.", end='\r')
    return pd.DataFrame(results).groupby(['ex_id', 'view'])['mae'].mean().reset_index()

def main():
    if not DATA_PATH.exists(): return
    
    print("🧹 Ex4(어브덕션) 제외 필터링 중...")
    all_df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    # Ex5(런지), Ex6(스쿼트)만 유지
    filtered_df = all_df[all_df['ex_id'].isin([5, 6])]
    
    # 1. Raw MediaPipe Calculation for filtered set
    raw_mae = filtered_df.groupby(['ex_id', 'view_type']).apply(lambda x: np.abs(x['mp_knee_angle']-x['gt_angle']).mean()).reset_index()
    raw_mae.columns = ['ex_id', 'view', 'raw_mae']
    raw_mae['ex_id'] = raw_mae['ex_id'].apply(lambda x: f"Ex{int(x)}")

    # 2. Run v7 comparison (Ex5, 6 only)
    print("\n🚀 v7 (ExtraTrees) 실험 (Ex5, 6 전용)...")
    v7_summary = run_loso_comparison(filtered_df, model_type="v7")
    v7_summary.rename(columns={'mae': 'v7_mae_only_56'}, inplace=True)

    # 3. Run v11 refinement & comparison (Ex5, 6 only)
    print("\n🚀 v11 (Physical Refined) 실험 (Ex5, 6 전용)...")
    refined_df = refine_bone_lengths(filtered_df)
    v11_summary = run_loso_comparison(refined_df, model_type="v11")
    v11_summary.rename(columns={'mae': 'v11_mae_only_56'}, inplace=True)

    # 4. Final Comparison
    report = pd.merge(raw_mae, v7_summary, on=['ex_id', 'view'])
    report = pd.merge(report, v11_summary, on=['ex_id', 'view'])
    
    # Improvement calculation relative to raw
    report['v7_impr (%)'] = (report['raw_mae'] - report['v7_mae_only_56']) / report['raw_mae'] * 100
    report['v11_impr (%)'] = (report['raw_mae'] - report['v11_mae_only_56']) / report['raw_mae'] * 100

    print("\n\n--- 🔥 v12 최종 성적표 (Ex4 완전 제외, Ex5(런지)/Ex6(스쿼트) 집중) ---")
    print(report.round(3).to_string(index=False))
    
    report.to_csv(SAVE_DIR / "v12_ex56_final_comparison.csv", index=False)
    print(f"\n✨ v12 최종 데이터 저장 완료: {SAVE_DIR}/v12_ex56_final_comparison.csv")

if __name__ == "__main__":
    main()
