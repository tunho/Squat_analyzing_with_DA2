import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
V7_CSV = PROJECT_ROOT / "outputs/ml_correction/v7_universal/v7_loso_results.csv"
V8_CSV = PROJECT_ROOT / "outputs/ml_correction/v8_experiments/cross_exercise_comparison.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v9_xgboost"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Lower-body focused Features (No Upper Body)
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

def run_xgboost_loso(df, train_ex_ids, test_ex_ids, exp_name):
    print(f"\n🚀 {exp_name} (XGBoost) LOSO validation in progress...")
    subjects = sorted(df['subject_id'].unique())
    results = []
    
    for i, test_sid in enumerate(subjects):
        train_df = df[(df['subject_id'] != test_sid) & (df['ex_id'].isin(train_ex_ids))]
        test_df = df[(df['subject_id'] == test_sid) & (df['ex_id'].isin(test_ex_ids))]
        
        if train_df.empty or test_df.empty: continue

        # Optimized XGBoost parameters for performance
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(train_df[FEATURES], train_df['gt_angle'])
        preds = model.predict(test_df[FEATURES])
        
        for eid in test_ex_ids:
            for view in ['front', 'side']:
                mask = (test_df['ex_id'] == eid) & (test_df['view_type'] == view)
                if mask.any():
                    mae = mean_absolute_error(test_df[mask]['gt_angle'], preds[mask])
                    results.append({"subject": test_sid, "ex_id": f"Ex{eid}", "view": view, "mae": mae})
        
        print(f"  [{i+1}/{len(subjects)}] Subject {test_sid} validated.", end='\r')
    
    res_df = pd.DataFrame(results)
    summary = res_df.groupby(['ex_id', 'view'])['mae'].mean().reset_index()
    summary.rename(columns={'mae': f'v9_mae'}, inplace=True)
    return summary

def main():
    if not DATA_PATH.exists():
        print(f"❌ Data not found: {DATA_PATH}")
        return
    
    all_df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    
    # 1. Raw MediaPipe Calculation
    all_df['raw_error'] = np.abs(all_df['mp_knee_angle'] - all_df['gt_angle'])
    raw_summary = all_df.groupby(['ex_id', 'view_type'])['raw_error'].mean().reset_index()
    raw_summary.columns = ['ex_id', 'view', 'raw_mae']
    raw_summary['ex_id'] = raw_summary['ex_id'].apply(lambda x: f"Ex{int(x)}")
    raw_summary = raw_summary[raw_summary['ex_id'].isin(['Ex5', 'Ex6'])]

    # 2. v7 Results (ExtraTrees)
    if V7_CSV.exists():
        v7_all = pd.read_csv(V7_CSV)
        v7_summary = v7_all.groupby(['exercise', 'view'])['mae'].mean().reset_index()
        v7_summary.columns = ['ex_id', 'view', 'v7_mae']
    else:
        v7_summary = pd.DataFrame(columns=['ex_id', 'view', 'v7_mae'])

    # 3. v8 Results (ET+GBR Ensemble)
    if V8_CSV.exists():
        v8_all = pd.read_csv(V8_CSV)
        # Use Joint_AI from v8 for fair comparison
        v8_summary = v8_all[v8_all['exp_name'] == 'Joint_AI'].groupby(['ex_id', 'view'])['mae'].mean().reset_index()
        v8_summary.columns = ['ex_id', 'view', 'v8_mae']
    else:
        v8_summary = pd.DataFrame(columns=['ex_id', 'view', 'v8_mae'])

    # 4. v9 (XGBoost) Execution
    v9_summary = run_xgboost_loso(all_df, [5, 6], [5, 6], "v9_Joint_XGBoost")

    # 5. Master Comparison
    master_df = pd.merge(raw_summary, v7_summary, on=['ex_id', 'view'], how='left')
    master_df = pd.merge(master_df, v8_summary, on=['ex_id', 'view'], how='left')
    master_df = pd.merge(master_df, v9_summary, on=['ex_id', 'view'], how='left')
    
    # Calculate Improvements (%)
    master_df['v9_improvement (%)'] = (master_df['raw_mae'] - master_df['v9_mae']) / master_df['raw_mae'] * 100
    
    print("\n\n--- 🔥 Master Performance Comparison (Raw, v7, v8, v9) ---")
    print(master_df.round(3).to_string(index=False))
    
    master_df.to_csv(SAVE_DIR / "v9_master_comparison.csv", index=False)
    print(f"\n✨ Master summary saved to: {SAVE_DIR}/v9_master_comparison.csv")

if __name__ == "__main__":
    main()
