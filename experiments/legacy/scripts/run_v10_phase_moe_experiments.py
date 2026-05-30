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
V9_CSV = PROJECT_ROOT / "outputs/ml_correction/v9_xgboost/v9_master_comparison.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v10_phase_moe"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Feature set (Lower-body focused)
FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2'
]

def run_v10_moe_loso(df, train_ex_ids, test_ex_ids, exp_name):
    print(f"\n🚀 {exp_name} (Phase-MOE) LOSO validation in progress...")
    subjects = sorted(df['subject_id'].unique())
    results = []
    
    # Velocity thresholds for phase separation
    THRES = 0.2
    
    for i, test_sid in enumerate(subjects):
        train_df = df[(df['subject_id'] != test_sid) & (df['ex_id'].isin(train_ex_ids))]
        test_df = df[(df['subject_id'] == test_sid) & (df['ex_id'].isin(test_ex_ids))]
        
        if train_df.empty or test_df.empty: continue

        # 1. Separate Training Data by Phase
        apex_train = train_df[np.abs(train_df['angle_vel_smooth']) < THRES]
        desc_train = train_df[train_df['angle_vel_smooth'] <= -THRES]
        asce_train = train_df[train_df['angle_vel_smooth'] >= THRES]

        # 2. Train Experts (using compact XGBoost parameters for speed)
        def create_expert(fold_df):
            if len(fold_df) < 20: return None
            m = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=6, n_jobs=-1, random_state=42)
            m.fit(fold_df[FEATURES], fold_df['gt_angle'])
            return m

        m_apex = create_expert(apex_train)
        m_desc = create_expert(desc_train)
        m_asce = create_expert(asce_train)

        # 3. Predict on Test Subject by frame-velocity
        test_preds = np.zeros(len(test_df))
        test_vel = test_df['angle_vel_smooth'].values
        
        # Mapping experts to frames
        for j in range(len(test_df)):
            vel = test_vel[j]
            feat = test_df.iloc[[j]][FEATURES]
            
            if abs(vel) < THRES and m_apex:
                test_preds[j] = m_apex.predict(feat)[0]
            elif vel <= -THRES and m_desc:
                test_preds[j] = m_desc.predict(feat)[0]
            elif vel >= THRES and m_asce:
                test_preds[j] = m_asce.predict(feat)[0]
            else:
                # Fallback to apex model or simple avg if others missing
                fallback = m_apex or m_desc or m_asce
                test_preds[j] = fallback.predict(feat)[0] if fallback else test_df.iloc[j]['mp_knee_angle']

        # 4. Record results per exercise/view
        for eid in test_ex_ids:
            for view in ['front', 'side']:
                mask = (test_df['ex_id'] == eid) & (test_df['view_type'] == view)
                if mask.any():
                    mae = mean_absolute_error(test_df[mask]['gt_angle'], test_preds[mask])
                    results.append({"subject": test_sid, "ex_id": f"Ex{eid}", "view": view, "mae": mae})
        
        print(f"  [{i+1}/{len(subjects)}] Subject {test_sid} validated.", end='\r')

    res_df = pd.DataFrame(results)
    summary = res_df.groupby(['ex_id', 'view'])['mae'].mean().reset_index()
    summary.rename(columns={'mae': 'v10_mae'}, inplace=True)
    return summary

def main():
    if not DATA_PATH.exists():
        print(f"❌ Data not found: {DATA_PATH}")
        return
    
    all_df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + ['gt_angle'])
    
    # 1. Load Raw, v7, v8, v9 baseline
    raw_summary = all_df.groupby(['ex_id', 'view_type'])['mp_knee_angle'].apply(lambda x: np.abs(x - all_df.loc[x.index, 'gt_angle']).mean()).reset_index()
    raw_summary.columns = ['ex_id', 'view', 'raw_mae']
    raw_summary['ex_id'] = raw_summary['ex_id'].apply(lambda x: f"Ex{int(x)}")
    raw_summary = raw_summary[raw_summary['ex_id'].isin(['Ex5', 'Ex6'])]

    # Helper function to get summary from CSV
    def load_summary(path, col_name, ex_col='ex_id', mae_col='mae'):
        if not Path(path).exists(): return pd.DataFrame(columns=['ex_id', 'view', col_name])
        df = pd.read_csv(path)
        if 'exp_name' in df.columns: df = df[df['exp_name'] == 'Joint_AI']
        s = df.groupby([ex_col, 'view'])[mae_col].mean().reset_index()
        s.columns = ['ex_id', 'view', col_name]
        return s

    v7_s = load_summary(V7_CSV, 'v7_mae', ex_col='exercise')
    v8_s = load_summary(V8_CSV, 'v8_mae')
    
    # v9 results come from the previously saved master_comparison if available, or its own csv
    v9_s = load_summary(PROJECT_ROOT / "outputs/ml_correction/v9_xgboost/v9_master_comparison.csv", 'v9_mae', mae_col='v9_mae')

    # 2. Run v10 MOE
    v10_s = run_v10_moe_loso(all_df, [5, 6], [5, 6], "v10_PhaseScale_MOE")

    # 3. Master Merge
    master = pd.merge(raw_summary, v7_s, on=['ex_id', 'view'], how='left')
    master = pd.merge(master, v8_s, on=['ex_id', 'view'], how='left')
    master = pd.merge(master, v9_s, on=['ex_id', 'view'], how='left')
    master = pd.merge(master, v10_s, on=['ex_id', 'view'], how='left')
    
    # Improvement calculation
    master['v10_impr (%)'] = (master['raw_mae'] - master['v10_mae']) / master['raw_mae'] * 100
    master['diff_v7_v10'] = master['v7_mae'] - master['v10_mae']

    print("\n\n--- 🔥 Final Master Comparison (v7 vs v10) ---")
    print(master.round(3).to_string(index=False))
    
    master.to_csv(SAVE_DIR / "v10_final_comparison.csv", index=False)
    print(f"\n✨ v10 results saved to: {SAVE_DIR}/v10_final_comparison.csv")

if __name__ == "__main__":
    main()
