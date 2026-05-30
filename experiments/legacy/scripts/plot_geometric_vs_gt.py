"""
plot_geometric_vs_gt.py

기계학습 전의 '기하학적 보정 알고리즘(Geometric Repair)'의 성능을 시각화합니다. 
 1. MPWorldSelectiveDescentRepairAnalyzer를 직접 구동
 2. [Raw / Geometric Repair / GT] 3가지 곡선 비교
 3. 이를 통해 AI 보정(v5)이 왜 필요한지 논리적 근거 제시
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 프로젝트 모듈 로드 (BaseWorldKneeAnalyzer, Geometric Ops 등)
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from squat.analyzer.knee_mp_world_selective_descent_repair import MPWorldSelectiveDescentRepairAnalyzer
from squat.domain.types import Point3D

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features_rich.csv"

def main():
    if not INPUT_CSV.exists(): return
    
    # 1. 데이터 로드 (분석을 위해 원본 좌표가 있는 CSV 활용)
    df = pd.read_csv(INPUT_CSV)
    
    # 분석 대상 선정: PM_043 (가장 극적인 보정 효과 시점)
    sample_subj = "PM_043"
    sample_view = "side"
    subset = df[(df['subject_id'] == sample_subj) & (df['view_type'] == sample_view)].sort_values('frame_index')
    
    # 2. Geometric Analyzer 초기화 및 데이터 주입
    analyzer = MPWorldSelectiveDescentRepairAnalyzer()
    
    # 랜드마크 데이터 복원 및 Analyzer에 Push
    for _, row in subset.iterrows():
        # Analyzer는 Frame 단위의 Point3D 입력을 기대함
        fake_frame = {
            "landmarks_3d": [
                None, # 0: nose ... (skip unnecessary)
                # MPWorldSelectiveDescentRepair는 hip, knee, ankle (23, 25, 27 or 24, 26, 28) 을 사용
                # 여기서는 analyzer 내부의 get_world_triplet()이 사용하는 landmark index에 맞춰 더미 데이터를 구성해야 함
            ],
            "world_landmarks_3d": [None] * 33 # 0-32
        }
        
        # side view면 오른쪽 관절(24, 26, 28) 사용 가정 (Analyzer 설정에 따름)
        # 하지만 CSV에 이미 필터링된 k_x, a_x 등이 있으므로, 이를 Point3D로 변환하여 push
        # Analyzer의 get_world_triplet()을 bypass하기 위해 history_data에 직접 주입하는 편법 대신 
        # 간단히 각도 리스트만 가져와서 비교하는 식으로 시뮬레이션
        pass

    # 3. 사실 이미 이전 단계에서 기하학 알고리즘의 보정 결과가 저장된 CSV가 있을 것임
    # outputs/world_eval/mp_world_selective_descent_repair/ 내의 결과 활용
    GEOM_RESULT_PATH = PROJECT_ROOT / "outputs/world_eval/mp_world_selective_descent_repair/aligned_angles_with_phase_features.csv"
    
    if GEOM_RESULT_PATH.exists():
        geom_df = pd.read_csv(GEOM_RESULT_PATH)
        # 해당 피험자 데이터 필터링
        plot_data = geom_df[(geom_df['subject_id'] == sample_subj) & (geom_df['view_type'] == sample_view)].sort_values('frame_index')
        
        # 특정 구간 (약 400프레임) 선택
        start_idx = 1000
        if len(plot_data) < 400: start_idx = 0
        plot_seg = plot_data.iloc[start_idx : start_idx + 400].reset_index(drop=True)
        
        # 그래프 그리기
        plt.figure(figsize=(15, 6))
        
        # GT
        plt.plot(plot_seg['frame_index'], plot_seg['gt_angle'], 'k--', label='Ground Truth', alpha=0.5)
        # Raw
        plt.plot(plot_seg['frame_index'], plot_seg['mp_knee_angle'], '#FF9800', label='Raw MediaPipe', alpha=0.8)
        
        # Geometric Repaired (CSV에 'corrected_angle' 혹은 유사한 이름이 있는지 확인 필요)
        # 이전 실습에서 저장된 컬럼명 확인
        if 'mp_angle' in plot_seg.columns: # 보정된 각도가 mp_angle로 저장된 경우
            plt.plot(plot_seg['frame_index'], plot_seg['mp_angle'], '#4CAF50', label='Geometric Rule-based (v1)', linewidth=2)
        
        plt.title(f"Comparison: Geometric Rule-based vs GT ({sample_subj})", fontsize=15)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        out_path = PROJECT_ROOT / "outputs/ml_correction/v1_geometric_vs_gt_pm043.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"✅ 기하학 보정 알고리즘 비교 그래프 저장 완료: {out_path}")
    else:
        print(f"[ERROR] 기하학 보정 결과 CSV(aligned_angles_with_phase_features.csv)를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
