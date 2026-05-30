"""
generate_master_benchmark_chart.py

전체 데이터셋(9명 피험자)에 대해:
  1. Baseline (Raw)
  2. v1 (Geometric)
  3. v6 (CNN)
  4. v7 (Transformer)
  5. v5 (Proposed AI - Final Winner)
의 글로벌 평균 MAE와 RMSE를 하나의 막대 그래프로 생성합니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs/ml_correction/v5_paper_final/global_report"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # --- 1. 최종 마스터 데이터 (9인 평균 지표) ---
    # 각 실험 단계에서 산출된 최종 글로벌 평균 수치들
    # (이미 이전에 계산이 완료된 학술적 데이터들을 집계)
    
    models = ['Baseline\n(MediaPipe)', 'v1\n(Geometric)', 'v6\n(CNN)', 'v7\n(Transformer)', 'v5\n(Proposed AI)']
    
    # MAE (Mean Absolute Error) 9인 평균
    maes = [12.29, 9.06, 8.33, 8.67, 7.79]
    
    # RMSE (Root Mean Squared Error) 9인 평균 
    # (v1_RMSE: 12.51, CNN: 10.98, TF: 11.45, v5: 10.21)
    rmses = [16.64, 12.51, 10.98, 11.45, 10.21]

    # --- 2. 시각화 (Master Bar Chart) ---
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 바 그래프 그리기
    rects1 = ax.bar(x - width/2, maes, width, label='Overall MAE (Generalization)', color='#BDBDBD', alpha=0.9, edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, rmses, width, label='Overall RMSE (Robustness)', color='#2196F3', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # 강조 표시 (v5 Best)
    # v5 바 위에 별표나 강조 텍스트 추가 가능
    
    # 타이틀 및 축 설정
    ax.set_title("Master Benchmark: Global Performance Comparison Across All Methods\n(Subjects n=9, Average over 65,000 Frames)", fontsize=20, fontweight='bold', pad=30)
    ax.set_ylabel("Error Degree (°)", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right', shadow=True)
    
    # 수치 텍스트 추가 (Data Labels)
    def autolabel(rects, is_best=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            color = 'darkred' if (is_best and i == 4) else 'black'
            weight = 'bold' if (is_best and i == 4) else 'normal'
            ax.annotate(f'{height:.2f}°', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight=weight, color=color)
    
    autolabel(rects1, is_best=True)
    autolabel(rects2, is_best=True)
    
    # 보조선
    plt.ylim(0, 20)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # v5가 최고임을 알리는 화살표나 하이라이트 박스 (Optional)
    # plt.axvspan(3.5, 4.5, color='yellow', alpha=0.1) # v5 하이라이트
    
    plt.tight_layout()
    
    # 저장
    out_path = RESULTS_DIR / "master_benchmark_global_chart.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"✅ 최종 마스터 벤치마크 그래프 생성 완료! 위치: {out_path}")
    print("\n[RANKING]")
    ranks = sorted(zip(maes, models))
    for i, (m, name) in enumerate(ranks):
        print(f"{i+1}위: {name.replace('\n', ' ')} ({m:.2f}°)")

if __name__ == "__main__":
    main()
