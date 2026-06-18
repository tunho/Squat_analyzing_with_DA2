# 단안 자세추정 기반 스쿼트 무릎각 보정

> MediaPipe 단안 3D 포즈의 **체계적 무릎각 오차**를 잔차 학습으로 보정하고,
> **4개 독립 데이터셋**에서 Leave-One-Subject-Out(LOSO) · Leave-One-Dataset-Out(LODO)로
> 검증한 연구 코드.

🇬🇧 English: [README.md](README.md)

---

## 한눈에

MediaPipe의 월드좌표 무릎각은 3D 정답 대비 **학습 가능한 체계적 오차**를 가진다.
`ExtraTreesRegressor`로 잔차 `gt_angle − mp_knee_angle`을 예측해 보정한다.

```
corrected_angle = mp_knee_angle + predicted_residual
```

4개 데이터셋(84명) 통합 시 무릎각 MAE가 **13.74° → 8.47° (+38.4%)**,
임상적으로 중요한 깊은 굴곡 구간(무릎 < 110°)에서는 **+43.5%** 개선된다.
80–130° 임계값 전 구간에서 개선율 차이가 3 %p 미만이라 **체리피킹이 아닌 강건한 결과**다.

## 데이터셋

| 데이터셋 | 역할 | 정답(GT) | 인원 |
|---|---|---|--:|
| **REHAB24-6** | 주 데이터 | 3D 마커 관절(`.npy`) | 9 |
| **FiT3D** | 외부 검증 | 3D mocap | 3 |
| **AIHub CrossFit** | 규모/도메인 | 3D 라벨 | 46 |
| **SUMediPose** | 골드 스탠다드 | Vicon 광학식 mocap | 26 |

정답은 모두 3D 관절 좌표이며, 추론 시에는 단안 MediaPipe 출력만 사용한다(배포 시 카메라만 필요).

## 핵심 결과

### 1. In-domain LOSO — 모든 데이터셋에서 일관되게 강함

| 데이터셋 | 원시 MAE | 보정 MAE | full | deep(<110°) |
|---|--:|--:|--:|--:|
| AIHub | 14.31° | 7.97° | **+44.3%** | +39.3% |
| SUMediPose (Vicon) | 12.65° | 7.26° | **+42.6%** | +53.5% |
| REHAB24-6 | 12.29° | 8.11° | **+34.0%** | +61.9% |
| FiT3D | 20.25° | 11.94° | **+41.0%** | +62.1% |

### 2. 베이스라인 비교 (pooled LOSO MAE °)

| 범위 | raw | savgol | affine | calib | linreg | **ours** |
|---|--:|--:|--:|--:|--:|--:|
| 전체 | 13.74 | 24.20 | 13.45 | 17.41 | 11.89 | **8.47** |
| deep <110° | 19.42 | 39.20 | 18.92 | 29.73 | 15.55 | **10.97** |

선형 잔차 베이스라인 대비 우위가 깊은 구간에서 *더 커진다*(원시 대비 +43.5% vs +19.9%, 2.2배).

### 3. 교차 데이터셋 일반화 — 경계를 정직하게 보고

Leave-One-Dataset-Out (3개로 학습 → 빠진 4번째 도메인 zero-shot 테스트):

| 빠진 도메인 | per-subject 개선율 | 95% CI |
|---|--:|--:|
| AIHub | −11.4% | [−14.0, −8.8] |
| SUMediPose | +21.5% | [+18.3, +24.7] |
| REHAB | +2.5% | [−4.8, +10.5] |
| FiT3D | +25.2% | [+16.4, +33.1] |
| **전체** | **+1.6%** | **[−2.2, +5.7]** |

zero-shot 도메인 전이는 전체적으로 **통계적으로 신뢰할 수 없다**(CI가 0 포함). 과장하지 않고
명시적으로 보고한다. 다만 깊은 굴곡 구간은 4개 중 3개 도메인에서 부분적으로 일반화된다
(AIHub +28%, FiT3D +35%, SUME +17%).

### 4. Few-shot 회복 — 타깃 1명이면 실패 도메인이 회복

타깃 도메인에서 단 **k=1명**만 학습에 추가(나머지로 테스트):

| 빠진 도메인 | k=0 (zero-shot) | k=1 | k=2 | 단독 LOSO |
|---|--:|--:|--:|--:|
| AIHub | −11.1 | **+13.0** | +20.3 | +44.3 |
| REHAB | +2.1 | **+15.9** | +19.9 | +34.0 |
| SUMediPose | +21.3 | +23.9 | +26.7 | +42.6 |
| FiT3D | +23.8 | +29.7 | +34.6 | +41.0 |

k≈2에서 거의 포화 — 새 도메인 적응에 필요한 *최소 타깃 감독량*을 정량화한다.

## 방법

```
영상(.mp4) + 3D GT 관절(.npy)
        │  scripts/paper_extract*.py   (MediaPipe 월드좌표, hip 중심, 길이 정규화)
        ▼
Feature CSV  [mp_knee_angle, gt_angle, k_x/y/z, a_x/y/z, vis, view_type, ...]
        │  scripts/paper_train_loso.py  (+시간/파생 피처 → ExtraTrees, LOSO)
        ▼
모델 번들(.pkl)
        │  scripts/paper_evaluate.py
        ▼
predictions.csv  [mp_knee_angle, corrected_angle, gt_angle, predicted_residual]
```

기본 피처셋 `enhanced_safe`(39개)는 현재/지연 hip-상대 좌표, 관절 가시성, 분절 길이
안정성, 각속도를 결합한다. 실시간 안전 변형 `enhanced_causal`(34개)은 오프라인 전용
정규화를 제거해 스트리밍에 쓸 수 있다.

## 저장소 구조

```
src/squat/            코어 라이브러리 (포즈 파이프라인, 기하, 분석기, 횟수 카운터)
scripts/              논문 파이프라인: 추출, LOSO/LODO 학습, few-shot, 베이스라인, 그림
experiments/paper/    집계 지표·요약 (대용량 데이터 CSV는 gitignore, 재생성 가능)
experiments/legacy/   초기 실험 스크립트 (이력 보존용)
docs/                 연구 노트, 논문 초안, 분석 계획서 (한국어)
```

> 대용량 원본 덤프, 프레임별 `predictions.csv`, 파생 feature CSV, `.pkl` 모델은
> **커밋하지 않는다**(용량 + 데이터셋 라이선스). 추출 스크립트로 완전히 재현 가능하다.

## 환경 설정

```bash
# 코어 런타임 (MediaPipe, OpenCV, NumPy)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 전체 파이프라인 (+ pandas, scikit-learn, matplotlib)
python -m venv .venv-paper && source .venv-paper/bin/activate
pip install -r requirements-paper.txt
```

## 라이선스 · 데이터

코드는 연구 목적이다. 데이터셋(REHAB24-6, FiT3D, AIHub CrossFit, SUMediPose)은
각자의 라이선스를 따르며 본 저장소에 포함되지 않는다 — 원 출처에서 받아야 한다.
