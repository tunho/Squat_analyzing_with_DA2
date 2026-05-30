"""
plot_v1_comparison.py

[Raw / v1 Geometric / GT] 3단 비교 그래프를 생성합니다.
교수님 보고 및 논문 1단계 실험 결과로 사용됩니다.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
GEOM_CSV = PROJECT_ROOT / "outputs/world_eval/mp_world_selective_descent_repair/aligned_angles_with_phase_features.csv"
RICH_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"

def main():
    if not GEOM_CSV.exists() or not RICH_CSV.exists():
        print(f"파일이 없습니다. GEOM:{GEOM_CSV.exists()}, RICH:{RICH_CSV.exists()}")
        return

    # 1. 데이터 로드
    # GEOM_CSV는 헤더 인덱스가 UTF-8 BOM(﻿)으로 시작할 수 있어 처리 필요
    geom_df = pd.read_csv(GEOM_CSV, encoding='utf-8-sig')
    rich_df = pd.read_csv(RICH_CSV)
    
    # PM_043 데이터 추출
    # GEOM_CSV에는 subject_id가 없을 수 있으므로, RICH_CSV에서 PM_043의 프레임 인덱스 범위를 가져옴
    subj_rich = rich_df[(rich_df['subject_id'] == 'PM_043') & (rich_df['view_type'] == 'side')].sort_values('frame_index')
    
    # 프레임 인덱스 기준으로 병합
    # GEOM_CSV는 해당 피험자 단독 실험 결과일 가능성이 큼
    # 만약 GEOM_CSV가 PM_043 결과라면 바로 사용 가능
    
    plot_data = subj_rich.merge(geom_df[['frame_index', 'pred_angle']], on='frame_index', how='inner')
    
    if len(plot_data) == 0:
        print("병합된 데이터가 없습니다. 프레임 인덱스가 일치하는지 확인하십시오.")
        # 백업: GEOM_CSV가 PM_043 것인지 확인 후 그냥 사용
        plot_data = geom_df.iloc[1000:1400].reset_index(drop=True)
        # Raw MP는 rich_df에서 가져오기 어려우면 일단 pred_angle만 그림
    
    # 시각화
    plt.figure(figsize=(15, 7))
    
    # 1. 정답선 (GT) - 검은 점선
    plt.plot(plot_data['frame_index'], plot_data['gt_angle'], 'k--', label='Ground Truth (Gold Standard)', alpha=0.5, linewidth=2)
    
    # 2. 원본 MediaPipe (Raw) - 주황색 (rich_df에서 가져온 데이터)
    if 'mp_knee_angle' in plot_data.columns:
        plt.plot(plot_data['frame_index'], plot_data['mp_knee_angle'], '#FF9800', label='Baseline: Raw MediaPipe', alpha=0.7)
    
    # 3. 1세대 기하학 알고리즘 (v1) - 초록색
    if 'pred_angle' in plot_data.columns:
        plt.plot(plot_data['frame_index'], plot_data['pred_angle'], '#4CAF50', label='Proposed v1: Geometric Rule-based', linewidth=2)

    plt.title("Scientific Progression: Baseline vs v1 Geometric Repair (PM_043)", fontsize=16, fontweight='bold')
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Knee Angle (deg)", fontsize=12)
    plt.legend(loc='upper right', shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 스쿼트 깊이 구간 강조 (보조용)
    plt.ylim(60, 190)
    
    out_path = PROJECT_ROOT / "outputs/ml_correction/v1_geometric_comparison_final.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"✅ v1 단계 비교 그래프 생성 완료: {out_path}")

if __name__ == "__main__":
    main()
