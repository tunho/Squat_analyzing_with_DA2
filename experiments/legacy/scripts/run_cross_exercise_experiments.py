import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v8_experiments"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

def run_experiment_loso(df, train_ex_ids, test_ex_ids, exp_name):
    print(f"\n🧪 {exp_name} 가동 중... (Ensemble Mode)")
    
    subjects = sorted(df['subject_id'].unique())
    results = []
    
    for i, test_sid in enumerate(subjects):
        train_df = df[(df['subject_id'] != test_sid) & (df['ex_id'].isin(train_ex_ids))]
        test_df = df[(df['subject_id'] == test_sid) & (df['ex_id'].isin(test_ex_ids))]
        
        if train_df.empty or test_df.empty: continue

        # Ensemble: ExtraTrees + GradientBoosting
        m1 = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        m2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        
        m1.fit(train_df[FEATURES], train_df['gt_angle'])
        m2.fit(train_df[FEATURES], train_df['gt_angle'])
        
        # 6:4 weighted average
        preds = (0.6 * m1.predict(test_df[FEATURES])) + (0.4 * m2.predict(test_df[FEATURES]))
        
        for eid in test_ex_ids:
            for view in ['front', 'side']:
                mask = (test_df['ex_id'] == eid) & (test_df['view_type'] == view)
                if mask.any():
                    mae = mean_absolute_error(test_df[mask]['gt_angle'], preds[mask])
                    results.append({"subject": test_sid, "ex_id": f"Ex{eid}", "view": view, "mae": mae})
                
    res_df = pd.DataFrame(results)
    summary = res_df.groupby(['ex_id', 'view'])['mae'].mean().reset_index()
    summary['exp_name'] = exp_name
    return summary

def main():
    if not DATA_PATH.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    
    # 실험 1: Joint (Ex5+6) -> Ex5, 6 평가 (LOSO)
    res_joint = run_experiment_loso(df, [5, 6], [5, 6], "Joint_AI")
    
    # 실험 2: Specific Ex5 -> Ex5 평가 (LOSO)
    res_spec5 = run_experiment_loso(df, [5], [5], "Specific_AI")
    
    # 실험 3: Specific Ex6 -> Ex6 평가 (LOSO)
    res_spec6 = run_experiment_loso(df, [6], [6], "Specific_AI")
    
    # 결과 통합
    final_comparison = pd.concat([res_joint, res_spec5, res_spec6])
    final_comparison.to_csv(SAVE_DIR / "cross_exercise_comparison.csv", index=False)
    
    print("\n--- 🔥 전이 학습 실험 결과 비교 (Joint vs Specific by View) ---")
    pivot_df = final_comparison.pivot_table(index=['ex_id', 'view'], columns='exp_name', values='mae', aggfunc='mean')
    print(pivot_df)
    print(f"\n✨ 상세 결과 저장됨: {SAVE_DIR}/cross_exercise_comparison.csv")

if __name__ == "__main__":
    main()
