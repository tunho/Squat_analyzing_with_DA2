import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Path settings
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature settings (same as v4_rich)
FEATURE_COLS = [
    "mp_knee_angle", "mp_hip_angle",
    "k_x", "k_y", "k_z", "a_x", "a_y", "a_z", "s_x", "s_y", "s_z",
    "h_vis", "k_vis", "a_vis", "s_vis",
    "hip_depth_norm", "hip_ankle_dist_norm",
    "knee_lag1", "knee_lag2"
]
TARGET_COL = "residual"

def prepare_data(df):
    df = df.sort_values(['subject_id', 'view_type', 'frame_index'])
    df['knee_lag1'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(1)
    df['knee_lag2'] = df.groupby(['subject_id', 'view_type'])['mp_knee_angle'].shift(2)
    df['residual'] = df['gt_angle'] - df['mp_knee_angle']
    df = df.dropna(subset=['knee_lag1', 'knee_lag2'])
    return df

def run_loso_and_collect_preds(df):
    subjects = sorted(df["subject_id"].unique())
    all_predictions = []

    print(f"Running LOSO for {len(subjects)} subjects...")

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj].copy()

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Using a slightly lighter model for faster plotting, but still high quality
        et = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

        et.fit(X_train_s, y_train)
        gbm.fit(X_train_s, y_train)

        pred_res = (0.6 * et.predict(X_test_s)) + (0.4 * gbm.predict(X_test_s))
        test_df["predicted_residual"] = pred_res
        test_df["corrected_angle"] = test_df["mp_knee_angle"] + pred_res
        
        all_predictions.append(test_df)
        print(f"  - Subject {test_subj} finished.")

    return pd.concat(all_predictions)

def plot_results(full_preds_df):
    plt.style.use('bmh')
    
    # 1. Scatter Plot (GT vs Corrected)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=full_preds_df, x='gt_angle', y='corrected_angle', hue='view_type', alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(full_preds_df['gt_angle'].min(), full_preds_df['corrected_angle'].min())
    max_val = max(full_preds_df['gt_angle'].max(), full_preds_df['corrected_angle'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title('Knee Angle Correction: GT vs Predicted (LOSO)', fontsize=15)
    plt.xlabel('Ground Truth Angle (°)', fontsize=12)
    plt.ylabel('Corrected Angle (°)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / 'loso_gt_vs_predicted_scatter.png', dpi=300)
    print(f"Saved scatter plot to {OUTPUT_DIR / 'loso_gt_vs_predicted_scatter.png'}")

    # 2. Time-series Plot (GT vs MP vs Corrected) for a sample subject (PM_008 front)
    sample_subj = 'PM_008'
    sample_df = full_preds_df[(full_preds_df['subject_id'] == sample_subj) & (full_preds_df['view_type'] == 'front')].head(500)
    
    if not sample_df.empty:
        plt.figure(figsize=(15, 6))
        plt.plot(sample_df['frame_index'], sample_df['gt_angle'], color='black', lw=2, label='GT Angle')
        plt.plot(sample_df['frame_index'], sample_df['mp_knee_angle'], color='red', alpha=0.6, label='Raw MediaPipe')
        plt.plot(sample_df['frame_index'], sample_df['corrected_angle'], color='blue', alpha=0.8, label='Corrected Angle')
        
        plt.title(f'Knee Angle Correction Time-series: Subject {sample_subj} (Front View)', fontsize=15)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Angle (°)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / 'loso_timeseries_sample.png', dpi=300)
        print(f"Saved time-series sample plot to {OUTPUT_DIR / 'loso_timeseries_sample.png'}")

    # 3. Correlation plot for MP vs GT
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=full_preds_df, x='gt_angle', y='mp_knee_angle', hue='view_type', alpha=0.3, s=10)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Correlation')
    plt.title('Raw MediaPipe vs GT (Before Correction)', fontsize=15)
    plt.xlabel('Ground Truth Angle (°)', fontsize=12)
    plt.ylabel('Raw MediaPipe Angle (°)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / 'raw_mp_vs_gt_scatter.png', dpi=300)
    print(f"Saved raw scatter plot to {OUTPUT_DIR / 'raw_mp_vs_gt_scatter.png'}")

def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    raw_df = pd.read_csv(INPUT_CSV)
    df = prepare_data(raw_df)
    
    full_preds = run_loso_and_collect_preds(df)
    plot_results(full_preds)
    
    # Save the full predictions for record
    # full_preds.to_csv(OUTPUT_DIR / 'loso_full_predictions.csv', index=False)
    # print(f"Saved full predictions to {OUTPUT_DIR / 'loso_full_predictions.csv'}")

if __name__ == "__main__":
    main()
