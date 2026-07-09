# 재현 가이드 (Reproduction Guide)

markerless 무릎각 잔차보정 벤치마크의 전 실험 재현 절차. 모든 학습 코드는 **seed 42** 고정.

> **재현 경로 2가지**
> - **(A) 원본 영상부터**: 데이터셋 취득 → 특징 추출(`paper_extract*`) → 학습/평가. 완전 재현이지만 데이터셋 접근 필요.
> - **(B) 특징 CSV부터**: 비식별 `features_*.csv`(관절 좌표/각도만, 영상 아님)만 있으면 보정 실험(①②③④) 전부 재현 가능. 영상 없이 프라이버시 안전.

---

## 1. 환경

```bash
python -m venv .venv-paper && source .venv-paper/bin/activate
pip install -r requirements-paper.txt   # torch·tabm·pytorch-frame 포함
export PYTHONPATH=src
```

추정기별 가중치/외부 repo는 대용량이라 비커밋 — 원저장소에서 별도 취득:
- **NLF**: isarandi/nlf (TorchScript) → `models_nlf/`
- **RTMW3D**: mmpose 1.3.2 → `mmpose_repo/`, `models_rtmw3d/`
- **MotionBERT**: 공식 ckpt → `MotionBERT/`, `models_motionbert/`
- **MediaPipe**: `pip` 설치로 충분

## 2. 데이터셋 (경로 `dataset/raw/`)

| 도메인 | 출처 | 접근 |
|---|---|---|
| REHAB24-6 | 공개 | 공개 다운로드 |
| FiT3D | 공개 | 공개 다운로드 |
| SUMediPose | Harvard Dataverse doi:10.7910/DVN/GRRROM | `scripts/download_sumedipose_a3.py` |
| AIHub | AIHub(한국) | 신청·승인 필요 |

인체 피험자 데이터라 **원본 영상은 재배포 불가**. 위 출처에서 직접 취득.

## 3. 특징 추출 (영상 → `experiments/paper/features_{도메인}_{추정기}.csv`)

```bash
cd scripts
# 추정기별 추출 스크립트 (MediaPipe=paper_extract, 나머지=estimator별)
PYTHONPATH=../src python paper_extract.py        --dataset ../dataset/raw/REHAB24-6 --exercise Ex6 --output ../experiments/paper/features_ex6_offstd.csv
PYTHONPATH=../src python paper_extract_nlf.py        ...   # NLF
PYTHONPATH=../src python paper_extract_rtmw3d.py     ...   # RTMW3D
PYTHONPATH=../src python paper_extract_motionbert.py ...   # MotionBERT
# AIHub·FiT3D 전용: paper_extract_aihub_est.py, paper_extract_fit3d_est.py, sumedipose_est.py
```

## 4. 실험 4종 (전부 seed 42)

### ① 도메인 내 (within-domain, LOSO)
```bash
# 트리/평활: baselines_loso.py,  신경망 5종: run_multiseed_within.py (seed 42)
python run_multiseed_within.py --gpus 0 --seeds 42
```
결과: `experiments/paper/multiseed_out/{보정기}/{도메인}_{추정기}/seed42/`

### ② pooled (5-fold GroupKFold)
```bash
python pooled_cv.py --corrector tabm --features <features> --folds 5 --seed 42
# 9종 × 4추정기 → pooled5_out/
```

### ③ 도메인 간 (LODO)
```bash
python gen_allcorr.py --corrector tabm --eval lodo --seed 42   # → lodo_base/
# ExtraTrees LODO: lodo_eval.py → lodo_{추정기}/lodo_metrics.csv
```

### ④ few-shot 회복
```bash
python gen_allcorr.py --corrector tabm --eval fewshot --seed 42
python fewshot_recovery.py ...   # 헤드라인 곡선
```

## 5. 통계·임상·그림
```bash
python corrector_stats.py     # Wilcoxon·부트스트랩 CI
python clinical_downstream.py # ICC·±5°·bias
python agreement_stats.py     # ICC 상세
python make_cube_heatmaps.py  # 실험별 히트맵 (①②③)
python make_bar_figures.py    # 보정기 순위·종합 막대
```

## 6. 재현성 메모
- 학습 코드는 `seed_util.set_all_seeds(42)` 로 random/numpy/torch/cuda + `cudnn.deterministic=True` 고정.
- **하드웨어 caveat**: GPU 신경망은 seed 고정해도 GPU/cuDNN 버전이 다르면 값이 미세하게 다를 수 있음(V100 vs 4070Ti 확인). 본 결과는 **4070Ti** 기준. 상세: `experiments/paper/재현성_재실행_대장.md`.
- sklearn(ExtraTrees·MLP)·평활기(SavGol·Kalman)는 하드웨어 무관 완전 결정적.
