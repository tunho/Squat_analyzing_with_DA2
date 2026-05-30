import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_CSV = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
LOSO_RESULTS_CSV = PROJECT_ROOT / "outputs/ml_correction/v7_universal/v7_loso_results.csv"
SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v7_universal/analysis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def run_analysis():
    print("📊 Raw vs v7 통합 분석 시작...")
    
    # 1. 원본 데이터 로드 (Raw MAE 계산용)
    df_raw = pd.read_csv(FEATURES_CSV)
    df_raw['raw_error'] = np.abs(df_raw['mp_knee_angle'] - df_raw['gt_angle'])
    
    raw_summary = df_raw.groupby(['ex_id', 'view_type'])['raw_error'].mean().reset_index()
    raw_summary.columns = ['exercise', 'view', 'raw_mae']
    raw_summary['exercise'] = raw_summary['exercise'].apply(lambda x: f"Ex{int(x)}")
    
    # 2. v7 LOSO 결과 로드
    df_v7 = pd.read_csv(LOSO_RESULTS_CSV)
    v7_summary = df_v7.groupby(['exercise', 'view'])['mae'].mean().reset_index()
    v7_summary.columns = ['exercise', 'view', 'v7_mae']
    
    # 3. 데이터 통합
    comp_df = pd.merge(raw_summary, v7_summary, on=['exercise', 'view'])
    comp_df['improvement'] = comp_df['raw_mae'] - comp_df['v7_mae']
    comp_df['improvement_pct'] = (comp_df['improvement'] / comp_df['raw_mae']) * 100
    
    # 4. 정면/측면 분리 저장
    comp_df[comp_df['view'] == 'front'].to_csv(SAVE_DIR / "v7_comparison_front.csv", index=False)
    comp_df[comp_df['view'] == 'side'].to_csv(SAVE_DIR / "v7_comparison_side.csv", index=False)
    comp_df.to_csv(SAVE_DIR / "v7_comparison_total.csv", index=False)
    
    print("\n--- [시점별 보정 성능 요약] ---")
    print(comp_df)
    
    # 5. 그래프 생성
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    plot_data = comp_df.melt(id_vars=['exercise', 'view'], value_vars=['raw_mae', 'v7_mae'], 
                            var_name='Model', value_name='MAE')
    plot_data['label'] = plot_data['exercise'] + " (" + plot_data['view'] + ")"
    
    palette = {'raw_mae': '#A0A0A0', 'v7_mae': '#2E86C1'}
    ax = sns.barplot(data=plot_data, x='label', y='MAE', hue='Model', palette=palette)
    
    plt.title("Generalization Performance: MediaPipe Raw vs. AI v7 (Universal)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Mean Absolute Error (Degrees)", fontsize=12)
    plt.xlabel("Exercise and View Type", fontsize=12)
    plt.xticks(rotation=15)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Raw MediaPipe', 'AI v7 (Universal)'], title="Target Model", frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}°', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "v7_performance_comparison_chart.png", dpi=300)
    print(f"\n✨ 그래프 저장 완료: {SAVE_DIR}/v7_performance_comparison_chart.png")

if __name__ == "__main__":
    run_analysis()
