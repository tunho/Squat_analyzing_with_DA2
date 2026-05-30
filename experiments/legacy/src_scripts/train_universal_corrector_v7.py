import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs/ml_correction/v7_universal/universal_features_complete.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "outputs/ml_correction/v7_universal/models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'mp_knee_angle', 'mp_hip_angle', 'leg_ratio',
    'k_x', 'k_y', 'k_z', 'a_x', 'a_y', 'a_z',
    'hip_depth_norm', 'hip_ankle_dist_norm',
    'angle_velocity', 'angle_vel_smooth', 'knee_lag1', 'knee_lag2',
    'ex_id'
]

def run_universal_loso():
    print("🚀 v7 통합 모델 LOSO 학습 시작...")
    if not DATA_PATH.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # NaN 정제
    df = df.dropna(subset=FEATURES + ['gt_angle'])
    subjects = sorted(df['subject_id'].unique())
    print(f"📊 피험자 수: {len(subjects)}명 | 전체 프레임 수: {len(df)}")
    
    results = []
    
    for i, test_sid in enumerate(subjects):
        train_df = df[df['subject_id'] != test_sid]
        test_df = df[df['subject_id'] == test_sid]
        
        if test_df.empty: continue

        X_train, y_train = train_df[FEATURES], train_df['gt_angle']
        X_test, y_test = test_df[FEATURES], test_df['gt_angle']
        
        # 모델 학습
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 예측
        preds = model.predict(X_test)
        
        # 운동별/시점별 성능 기록
        for ex_id in [4, 5, 6]:
            for view in ['front', 'side']:
                mask = (test_df['ex_id'] == ex_id) & (test_df['view_type'] == view)
                if mask.any():
                    mae = mean_absolute_error(y_test[mask], preds[mask])
                    results.append({
                        "test_subject": test_sid,
                        "exercise": f"Ex{ex_id}",
                        "view": view,
                        "mae": mae
                    })
        
        print(f"  [{i+1}/{len(subjects)}] Subject {test_sid} 검증 완료. (진행률: {(i+1)/len(subjects)*100:.1f}%)")

    # 결과 분석 및 저장
    res_df = pd.DataFrame(results)
    res_df.to_csv(PROJECT_ROOT / "outputs/ml_correction/v7_universal/v7_loso_results.csv", index=False)
    
    summary = res_df.groupby(['exercise', 'view'])['mae'].mean().reset_index()
    print("\n--- [v7 통합 모델 운동/시점별 평균 MAE] ---")
    print(summary)
    
    overall_summary = res_df.groupby('exercise')['mae'].mean().reset_index()
    print("\n--- [v7 통합 모델 운동별 전체 평균 MAE] ---")
    print(overall_summary)
    
    total_avg = res_df['mae'].mean()
    print(f"\n🔥 전체 통합 평균 MAE: {total_avg:.4f}")
    
    # 최종 배포용 통합 모델 저장
    final_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(df[FEATURES], df['gt_angle'])
    joblib.dump(final_model, MODEL_SAVE_DIR / "universal_corrector_v7.pkl")
    print(f"\n✨ v7 통합 모델 저장 완료: {MODEL_SAVE_DIR}/universal_corrector_v7.pkl")

if __name__ == "__main__":
    run_universal_loso()
