import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
from pathlib import Path

def generate_plots(metrics_path, save_dir):
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found.")
        return
        
    df = pd.read_csv(metrics_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall Performance by View (Bar Chart)
    view_summary = df.groupby('view')[['mae_raw', 'mae_model']].mean().reset_index()
    view_summary_melted = view_summary.melt(id_vars='view', var_name='Type', value_name='MAE')
    view_summary_melted['Type'] = view_summary_melted['Type'].map({'mae_raw': 'Raw MediaPipe', 'mae_model': 'Final Model (v13)'})
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=view_summary_melted, x='view', y='MAE', hue='Type', palette='viridis')
    plt.title('Knee Angle Estimation Performance: Final Model vs Raw', fontsize=15)
    plt.ylabel('Mean Absolute Error (degrees)', fontsize=12)
    plt.xlabel('View Type', fontsize=12)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')
    
    
    plt.savefig(save_dir / "mae_comparison_by_view.png", dpi=300)
    plt.close()
    
    id_col = 'subject' if 'subject' in df.columns else 'video_id'
    
    # 2. Subject-wise breakdown for Front View
    front_df = df[df['view'] == 'front'].sort_values(id_col)
    if not front_df.empty:
        front_melted = front_df.melt(id_vars=id_col, value_vars=['mae_raw', 'mae_model'], var_name='Type', value_name='MAE')
        front_melted['Type'] = front_melted['Type'].map({'mae_raw': 'Raw', 'mae_model': 'Model'})
        
        plt.figure(figsize=(14, 6))
        sns.barplot(data=front_melted, x=id_col, y='MAE', hue='Type', palette='magma')
        plt.title('Subject-wise MAE Comparison (Front View)', fontsize=15)
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        plt.savefig(save_dir / "subject_wise_front.png", dpi=300)
        plt.close()

    # 3. Subject-wise breakdown for Side View
    side_df = df[df['view'] == 'side'].sort_values(id_col)
    if not side_df.empty:
        side_melted = side_df.melt(id_vars=id_col, value_vars=['mae_raw', 'mae_model'], var_name='Type', value_name='MAE')
        side_melted['Type'] = side_melted['Type'].map({'mae_raw': 'Raw', 'mae_model': 'Model'})
        
        plt.figure(figsize=(14, 6))
        sns.barplot(data=side_melted, x=id_col, y='MAE', hue='Type', palette='magma')
        plt.title('Subject-wise MAE Comparison (Side View)', fontsize=15)
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        plt.savefig(save_dir / "subject_wise_side.png", dpi=300)
        plt.close()
        
    print(f"📊 Plots saved to {save_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        m_path = Path(sys.argv[1])
        s_dir = Path(sys.argv[2])
    else:
        PROJECT_ROOT = Path("/home/lee/exe_est")
        m_path = PROJECT_ROOT / "outputs/ml_correction/v13_mega_loso_comparison/v13_loso_comparison_metrics.csv"
        s_dir = PROJECT_ROOT / "outputs/ml_correction/v13_mega_loso_comparison/plots"
    
    generate_plots(m_path, s_dir)
