import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_CSV = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
CROSS_EXP_CSV = PROJECT_ROOT / "outputs/ml_correction/v8_experiments/cross_exercise_comparison.csv"
SAVE_PATH = PROJECT_ROOT / "outputs/ml_correction/v8_experiments/final_mae_report.csv"

def generate_report():
    print("📋 최종 학술 통합 리포트 생성 중 (View-Specific: Front vs Side)...")
    
    if not FEATURES_CSV.exists() or not CROSS_EXP_CSV.exists():
        print("❌ 필요한 데이터 파일이 없습니다.")
        return

    # 1. Raw MAE 계산 (View 포함)
    raw_df = pd.read_csv(FEATURES_CSV)
    raw_df['error'] = np.abs(raw_df['mp_knee_angle'] - raw_df['gt_angle'])
    raw_summary = raw_df.groupby(['ex_id', 'view_type'])['error'].mean().reset_index()
    raw_summary['ex_id'] = raw_summary['ex_id'].apply(lambda x: f"Ex{int(x)}")
    raw_summary.columns = ['Exercise', 'View', 'Raw MediaPipe']
    
    # 2. 실험 결과 로드 (View 포함)
    exp_df = pd.read_csv(CROSS_EXP_CSV)
    exp_pivot = exp_df.pivot_table(index=['ex_id', 'view'], columns='exp_name', values='mae', aggfunc='mean').reset_index()
    exp_pivot.columns = ['Exercise', 'View', 'Joint AI', 'Specific AI']
    
    # 3. 통합
    final_report = pd.merge(raw_summary, exp_pivot, on=['Exercise', 'View'])
    
    # 컬럼 재배치
    final_report = final_report[['Exercise', 'View', 'Raw MediaPipe', 'Specific AI', 'Joint AI']]
    
    # 4. 개선율 계산
    final_report['Improvement (%)'] = ((final_report['Raw MediaPipe'] - final_report['Joint AI']) / final_report['Raw MediaPipe']) * 100
    
    # Ex5, Ex6 필터링
    final_report = final_report[final_report['Exercise'].isin(['Ex5', 'Ex6'])]
    
    print("\n" + "="*95)
    print("✨ [최종 학술 데이터 통합 리포트: Lunge & Abduction by View]")
    print("="*95)
    print(final_report.to_string(index=False))
    print("="*95)
    
    final_report.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ 리포트 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    generate_report()
