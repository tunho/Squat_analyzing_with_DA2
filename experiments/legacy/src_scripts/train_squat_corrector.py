"""
train_squat_corrector.py

extract_squat_features.py로 생성된 CSV를 로드하여:
 1. Leave-One-Subject-Out (LOSO) 교차검증으로 일반화 성능 검증
 2. 7:2 Hold-out으로 최종 모델 학습 및 저장

모델: GradientBoostingRegressor (sklearn)
입력 피처: mp_angle, hip_depth_norm, hip_ankle_dist_norm
출력: corrected_angle (GT에 가까운 보정 각도)

사용법:
  python src/train_squat_corrector.py

출력:
  outputs/ml_correction/loso_results.csv
  outputs/ml_correction/correction_model.pkl
  outputs/ml_correction/comparison_table.png
  outputs/ml_correction/feature_importance.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "outputs/ml_correction/all_subjects_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/ml_correction"

# 학습에 사용할 피처
FEATURE_COLS = ["mp_angle", "hip_depth_norm", "hip_ankle_dist_norm"]
TARGET_COL = "gt_angle"

# 7:2 train/test split (피험자 기준)
TRAIN_SUBJECTS = ["PM_008", "PM_022", "PM_029", "PM_038", "PM_043", "PM_105", "PM_113"]
TEST_SUBJECTS = ["PM_118", "PM_126"]

# 모델 하이퍼파라미터
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "subsample": 0.8,
    "random_state": 42,
}


def compute_phase_mae(df: pd.DataFrame, pred_col: str) -> dict:
    """단계별 MAE를 계산."""
    result = {}
    if "gt_phase" in df.columns:
        for phase in ["top", "descent", "bottom", "ascent"]:
            mask = df["gt_phase"] == phase
            if mask.any():
                result[f"mae_{phase}"] = float(mean_absolute_error(
                    df.loc[mask, TARGET_COL], df.loc[mask, pred_col]
                ))
    return result


def run_loso_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-One-Subject-Out 교차검증 실행."""
    subjects = sorted(df["subject_id"].unique())
    loso_rows = []

    print("\n" + "="*60)
    print("LEAVE-ONE-SUBJECT-OUT (LOSO) 교차검증")
    print("="*60)

    for test_subj in subjects:
        train_df = df[df["subject_id"] != test_subj]
        test_df = df[df["subject_id"] == test_subj].copy()

        X_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df[TARGET_COL].to_numpy()
        X_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df[TARGET_COL].to_numpy()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        mae_raw = float(mean_absolute_error(y_test, test_df["mp_angle"].to_numpy()))
        mae_corrected = float(mean_absolute_error(y_test, y_pred))
        improvement = mae_raw - mae_corrected

        row = {
            "test_subject": test_subj,
            "train_subjects": ",".join([s for s in subjects if s != test_subj]),
            "n_test_frames": len(test_df),
            "mae_raw_mp": mae_raw,
            "mae_corrected": mae_corrected,
            "improvement_deg": improvement,
            "improvement_pct": improvement / mae_raw * 100 if mae_raw > 0 else 0,
        }
        loso_rows.append(row)

        print(f"  [{test_subj}] Raw MAE: {mae_raw:.2f}° → Corrected MAE: {mae_corrected:.2f}° "
              f"(개선: {improvement:.2f}°, {row['improvement_pct']:.1f}%)")

    loso_df = pd.DataFrame(loso_rows)
    print("\n" + "-"*60)
    print(f"  LOSO 평균 Raw MAE:       {loso_df['mae_raw_mp'].mean():.2f}°")
    print(f"  LOSO 평균 Corrected MAE: {loso_df['mae_corrected'].mean():.2f}°")
    print(f"  LOSO 평균 개선:          {loso_df['improvement_deg'].mean():.2f}°"
          f" ({loso_df['improvement_pct'].mean():.1f}%)")
    print("="*60)

    return loso_df


def train_final_model(df: pd.DataFrame) -> tuple:
    """7:2 Hold-out으로 최종 모델 학습."""
    train_df = df[df["subject_id"].isin(TRAIN_SUBJECTS)]
    test_df = df[df["subject_id"].isin(TEST_SUBJECTS)].copy()

    if train_df.empty or test_df.empty:
        print("[WARNING] 훈련 또는 테스트 피험자 데이터가 없습니다!")
        all_subjects = df["subject_id"].unique().tolist()
        n_train = int(len(all_subjects) * 0.7)
        train_subjects = all_subjects[:n_train]
        test_subjects = all_subjects[n_train:]
        train_df = df[df["subject_id"].isin(train_subjects)]
        test_df = df[df["subject_id"].isin(test_subjects)].copy()

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    test_df["corrected_angle"] = y_pred

    mae_raw = float(mean_absolute_error(y_test, test_df["mp_angle"].to_numpy()))
    mae_corrected = float(mean_absolute_error(y_test, y_pred))

    print("\n" + "="*60)
    print("7:2 HOLD-OUT 최종 결과")
    print("="*60)
    print(f"  훈련 피험자: {train_df['subject_id'].unique().tolist()}")
    print(f"  테스트 피험자: {test_df['subject_id'].unique().tolist()}")
    print(f"  훈련 프레임: {len(train_df)}, 테스트 프레임: {len(test_df)}")
    print(f"\n  Raw MediaPipe MAE:  {mae_raw:.2f}°")
    print(f"  ML 보정 후 MAE:     {mae_corrected:.2f}°")
    print(f"  개선 폭:            {mae_raw - mae_corrected:.2f}° ({(mae_raw - mae_corrected) / mae_raw * 100:.1f}%)")

    # 피처 중요도
    print(f"\n  피처 중요도:")
    for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
        print(f"    {feat}: {imp:.3f}")

    print("="*60)

    return model, scaler, test_df, mae_raw, mae_corrected


def plot_comparison(test_df: pd.DataFrame, loso_df: pd.DataFrame, mae_raw: float, mae_corrected: float):
    """논문용 비교 시각화 생성."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ML-Based Knee Angle Correction: Results", fontsize=14, fontweight="bold")

    # 1. LOSO 피험자별 MAE 비교
    ax1 = axes[0]
    x = np.arange(len(loso_df))
    width = 0.35
    bars1 = ax1.bar(x - width/2, loso_df["mae_raw_mp"], width, label="Raw MediaPipe", color="#E87D00", alpha=0.85)
    bars2 = ax1.bar(x + width/2, loso_df["mae_corrected"], width, label="ML Corrected", color="#2196F3", alpha=0.85)
    ax1.set_xlabel("Subject ID")
    ax1.set_ylabel("MAE (degrees)")
    ax1.set_title("LOSO-CV: Per-Subject MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(loso_df["test_subject"].str.replace("PM_", "S"), rotation=30)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 레이블 표시
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    # 2. 전체 요약 (방법론 비교)
    ax2 = axes[1]
    methods = ["Raw\nMediaPipe", "3-Stage\nGeometric", "ML\nCorrected"]
    # 3-Stage 결과는 이전 실험에서 얻은 7.93 사용
    maes = [mae_raw, 7.93, mae_corrected]
    colors = ["#E87D00", "#FF9800", "#2196F3"]
    bars = ax2.bar(methods, maes, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("MAE (degrees)")
    ax2.set_title("Method Comparison: Overall MAE")
    ax2.set_ylim(0, max(maes) * 1.3)
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}°', ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis="y", alpha=0.3)

    # 3. 산점도: Raw vs Corrected vs GT
    ax3 = axes[2]
    sample = test_df.sample(min(2000, len(test_df)), random_state=42)
    ax3.scatter(sample["gt_angle"], sample["mp_angle"],
                alpha=0.3, s=5, color="#E87D00", label="Raw MediaPipe")
    ax3.scatter(sample["gt_angle"], sample["corrected_angle"],
                alpha=0.3, s=5, color="#2196F3", label="ML Corrected")
    lim = [sample["gt_angle"].min() - 5, sample["gt_angle"].max() + 5]
    ax3.plot(lim, lim, 'k--', linewidth=1, label="Perfect Prediction")
    ax3.set_xlabel("GT Angle (degrees)")
    ax3.set_ylabel("Predicted Angle (degrees)")
    ax3.set_title("GT vs Prediction Scatter")
    ax3.legend(markerscale=3)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "comparison_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 비교 그래프 저장: {out_path}")


def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] 피처 CSV 파일이 없습니다: {INPUT_CSV}")
        print("먼저 'python src/extract_squat_features.py' 를 실행하세요.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"✅ 데이터 로드 완료: {len(df)} 프레임, {df['subject_id'].nunique()} 명 피험자")
    print(f"   피험자 목록: {sorted(df['subject_id'].unique())}")

    # 1. LOSO 교차검증
    loso_df = run_loso_cv(df)
    loso_df.to_csv(OUTPUT_DIR / "loso_results.csv", index=False)
    print(f"\n  LOSO 결과 저장: {OUTPUT_DIR / 'loso_results.csv'}")

    # 2. 7:2 Hold-out 최종 모델 학습
    model, scaler, test_df, mae_raw, mae_corrected = train_final_model(df)

    # 3. 모델 저장
    model_bundle = {"model": model, "scaler": scaler, "features": FEATURE_COLS}
    with open(OUTPUT_DIR / "correction_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n  모델 저장: {OUTPUT_DIR / 'correction_model.pkl'}")

    # 4. 논문용 시각화
    plot_comparison(test_df, loso_df, mae_raw, mae_corrected)

    print(f"\n{'='*60}")
    print("🎉 모든 완료! 최종 요약:")
    print(f"  Raw MediaPipe MAE:  {mae_raw:.2f}°")
    print(f"  3-Stage Geometric:  ~7.93°  (이전 실험)")
    print(f"  ML Corrected MAE:   {mae_corrected:.2f}°")
    print(f"  LOSO 평균 MAE:      {loso_df['mae_corrected'].mean():.2f}°")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
