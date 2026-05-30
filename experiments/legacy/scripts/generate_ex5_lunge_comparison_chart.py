import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/ex5_evaluation_results.csv"

def main():
    if not CSV_PATH.exists():
        print(f"❌ 데이터 파일이 없습니다: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("❌ 결과가 비어있습니다.")
        return

    # 그래프 생성
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(df))
    width = 0.35
    
    # 1. Raw MAE
    plt.bar(x - width/2, df['mae_raw'], width, label='Baseline (Raw MediaPipe)', color='#FFCC80', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # 2. V5 Corrected MAE
    plt.bar(x + width/2, df['mae_v5'], width, label='Proposed AI (v5 - Squat Expert)', color='#90CAF9', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # 설정
    plt.title("Performance Transfer: Squat-Trained v5 Model on Lunge (Ex5) Dataset", fontsize=18, fontweight='bold', pad=25)
    plt.ylabel("Mean Absolute Error (°)", fontsize=14)
    plt.xlabel("Video Samples (Lunge Exercise)", fontsize=14)
    plt.xticks(x, df['video'], rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=12, loc='upper right', shadow=True)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 수치 텍스트 추가
    for i, row in df.iterrows():
        plt.text(i - width/2, row['mae_raw'] + 0.1, f"{row['mae_raw']:.1f}", ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, row['mae_v5'] + 0.1, f"{row['mae_v5']:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='blue')

    # 평균 개선 수치 표시
    avg_raw = df['mae_raw'].mean()
    avg_v5 = df['mae_v5'].mean()
    plt.axhline(avg_raw, color='#FB8C00', linestyle=':', alpha=0.5)
    plt.axhline(avg_v5, color='#1976D2', linestyle=':', alpha=0.5)
    
    plt.text(len(df)-0.5, avg_raw + 0.2, f"Avg Raw: {avg_raw:.2f}°", color='#E65100', fontweight='bold', ha='right')
    plt.text(len(df)-0.5, avg_v5 + 0.2, f"Avg v5: {avg_v5:.2f}°", color='#0D47A1', fontweight='bold', ha='right')

    plt.tight_layout()
    
    out_path = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/ex5_lunge_evaluation_chart.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"✅ Ex5(런지) 평가 차트 생성 완료: {out_path}")

if __name__ == "__main__":
    main()
