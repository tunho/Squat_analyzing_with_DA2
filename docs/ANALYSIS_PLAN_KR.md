# 분석 실행 계획 (에이전트용) — 스쿼트 무릎각 보정 전이분석

전제: 보정 방법 추가개발(③) 보류. **현재 결과 분석만.** 목표: 중급 확실 / 상급 도전.
상태범례: [ ]대기 [~]진행 [x]완료 [!]사용자결정필요

---

## 보유 자산 (입력)
- raw 덤프(world33+image33+GT, 재파생 가능): `experiments/paper/{aihub_raw_dump, sumedipose_raw_dump}/`
- 피처 CSV(causal,calib15): `features_{aihub_squat_causal[_ds6], sumedipose_causal, ex6_causal, fit3d_causal}.csv`
- pooled: `features_pooled4_balanced.csv`
- 결과: `outputs/paper/{aihub_loso_causal, sumedipose_loso, pooled4_balanced_loso}`, `experiments/paper/xval_*`
- 파생 스크립트: `aihub_features_from_dump.py`(--calib-frames,--frame-step), `paper_train_loso.py`(fold별 진행출력 추가됨), `paper_evaluate.py`
- 캘리브: SUMediPose `dataset/raw/sumedipose_a3/calibration/`(pairwise R,T .npz + tab), FiT3D `dataset/raw/fit3d/.../camera_parameters/*/squat.json`(R,T absolute)
- 미보유: FiT3D/REHAB MediaPipe 2D(=PnP용), AIHub/REHAB 카메라 캘리브(PnP로 추정)

---

## [x] 확정된 프로토콜 (2026-06-04)
- **정규화: offline(calib-0)** ← online(causal) 폐기. 이유: 실제 논문엔 offline이 더 깨끗·고성능·첫15프레임 가정 제거·실시간 과장 reject 회피. (causal은 "실시간 변형 가능" 보조 언급만)
- **피처셋: enhanced_safe(39)** ← offline이라 causal 제약 풀려 깊이/시퀀스 피처 추가.
- **fps: 30fps 유지**(다운샘플 X) — 작은셋 보존, temporal 충실.
- **밸런싱: 인당 strided 캡 3000** (전체 영상 균등샘플 → 동작·카메라 보존, head 버그 수정). 단독·pooled 동일.
- 표준 입력: `features_{*}_offstd.csv`, `features_pooled4_offstd.csv`.
- (검증중) time-aware(deg/s temporal) 변형도 비교 — pooled의 가변stride 시간스케일 흠 영향 확인용.

---

## A. [x] 데이터 프로토콜 통일 (완료)
- offline+enhanced_safe+30fps+strided캡3000 으로 4셋 표준화·LOSO 완료.
- **결과 (a) offline+enhanced_safe**: AIHub단독 +44.3%, SUMediPose단독 +42.6%, pooled4 +38.4%(전원개선, deep 전부+). → `outputs/paper/{aihub,sumedipose,pooled4}_offstd_loso`
- (b) time-aware 비교 진행중(deg/s) → 차이 미미하면 (a) 채택, 크면 (b).

## B. [x] 베이스라인 비교 완료 ★중급 gate
- 최종(offline+enhanced_safe): **ours +38.4% >> linreg +13.5% > affine +2.1% > savgol≈0 > calib −26.7%**. 2.8배 → 비선형 핵심. 중급 gate 통과. → `outputs/paper/baselines/pooled4_offstd_baselines.csv`
- (a)vs(b) temporal: ±0.3% 동일 → 프레임기반(a) 채택, time-aware는 robustness 언급.
- 의존: A
- 작업: 신규 `scripts/baselines_loso.py` 작성. 입력 feature CSV, 출력 방법별 LOSO MAE.
  - 비교군(동일 LOSO): ① raw ② 시간평활(savgol/이동평균, mp_knee_angle 시퀀스별) ③ 전역 선형보정(train서 a·b fit) ④ per-subject 캘리브(test 초기K프레임 평균오프셋) ⑤ 선형회귀(동일 enhanced_causal 피처) ⑥ ours(ExtraTrees)
  - 시퀀스 그룹 = (subject_id, view_type[, camera]) — paper_train_loso의 prepare 재사용.
- 산출: `outputs/paper/baselines/baseline_compare.csv` (방법×데이터셋×{전체,deep} MAE/개선율)
- 수용기준: ours가 모든 베이스라인 상회(특히 deep). 표 완성.

## C. [x] 통계 완료
- 평균개선 38.4% 95%CI[36.8,39.9], 84/84개선, ours vs linreg Wilcoxon p=1.7e-15. → per_subject_stats.csv
- 의존: B
- 작업: `scripts/stats_ci.py`. per-subject 개선율 부트스트랩 95% CI + ours vs 최고베이스라인 paired test(Wilcoxon).
- 산출: CI 동반 표(B 표에 컬럼 추가), p값.

## D. [x] 카메라 기하 분석 (corrected 중심으로 핵심 완료) ★상급
- **핵심결과(균등화)**: 카메라별 raw→corrected 카메라간 오차 std 감소 — AIHub 4.46→1.98°, SUMediPose 3.73→1.14°, REHAB 1.30→0.75°, FiT3D 6.26→3.41° (4셋 전부). 최악카메라 최대개선.
- raw(메커니즘)+corrected(방법robustness)+전이매트릭스(안본기하 실패)+z시프트로 카메라기하 논증 완결.
- DLT 거리/높이 정량회귀 = underpowered(n=6/셋, 등거리리그) → 생략(또는 cross 20캠 풀링 선택). SUMediPose DLT 카메라위치 복원은 검증됨(distance 40~44, 거리-오차 r=0.57 n부족).
## D-old. 카메라 기하 분석 (원계획)
### D1. [ ] SUMediPose 통제실험
- 작업: 제공 extrinsics(pairwise R,T)를 **기준카메라로 체인→절대 카메라 위치** 복원. external_calibration.tab의 main/pair 그래프 사용. Vicon 좌표계와 정합(또는 카메라간 상대 거리·높이만으로도 분석).
  - 카메라별: 피험자 중심까지 **거리·높이·azimuth** 계산.
  - 카메라별 raw 오차(보유: C1 12.6/C2 8.6/C3 12.7/C4 12.2/C5 9.4/C6 19.7) 와 상관.
- 산출: `outputs/paper/camera_geom/sumedipose_geom_vs_error.csv` + 산점도.
- 수용기준: 거리/높이↔오차 상관계수·유의성.
### D2. [ ] 교란변수 분리
- 작업: 오차 ~ 거리+높이+azimuth+관절가시성 다중회귀/부분상관. 어느 변수가 주범인지.
- 산출: 변수 기여도 표.
### D3. [ ] cross-dataset PnP 추정 (4셋 통일)
- 선행: FiT3D/REHAB **MediaPipe 2D 추출**(영상 로컬: dataset/raw/{fit3d,REHAB24-6}). AIHub/SUMediPose는 덤프 p*(2D) 사용.
- 작업: `scripts/pnp_camera.py`. GT3D+2D → cv2.solvePnP → 카메라 거리·높이. **검증: SUMediPose/FiT3D 제공 extrinsics vs PnP → 오차 보고**.
- 산출: `outputs/paper/camera_geom/xdataset_geom.csv`(4셋 거리·높이 + PnP검증오차)
- 한계: AIHub/REHAB 추정(불확실성 명시), REHAB 노이즈 큼.
### D4. [ ] 종합 연결
- 작업: 카메라기하 ↔ z-시프트(보유: a_z AIHub+0.28/REHAB+0.08) ↔ 전이매트릭스(보유) 일관성 정리.
- 산출: 종합 그림(기하 유사→전이○).

## E. [x] 코호트 완료
- AIHub 46(나이22~50·키152~190·성19/27), SUMediPose 26(나이21~26·키158~193·M15F11·스킨톤1~5). 합산 84명 다양성 확보.
- 작업: AIHub(annotation.json: 연령22~50·성별·키, 보유분석 재실행) + SUMediPose(`subject_info.tab`: 스킨톤·나이·성별·키). FiT3D/REHAB 가용분.
- 산출: `outputs/paper/cohort.csv` + 표.

## F. [x] 세부성능 완료
- view: AIHub front+41/side+33, REHAB front+19/side+45. deep<110 전셋+39~54%.
## F-old. 세부 성능 (대부분 보유, 정리만)
- 깊은굴곡<110°·뷰(front/side)·스쿼트변형별 개선율 표 정리(기존 결과 취합).

## G. [ ] 그림·작성
- 그림: 베이스라인비교(±CI), 데이터셋별/deep 개선, 전이매트릭스, **카메라기하↔오차 산점도**, 각도 trace(raw/corr/GT).
- 작성: PAPER_OUTLINE_KR.md 흐름대로.

---

## 실행 순서
A → B → C → E → F → (D1→D2→D3→D4) → G
- **중급 목표**: A·B·C·E·F·G (D 생략 가능)
- **상급 목표**: + D 전체

## 다음 행동
1. [!] DECISION-1,2 사용자 확인 → A 착수.
2. A 완료 후 B(baselines_loso.py) 작성·실행.

관련: `PAPER_OUTLINE_KR.md`(데이터·결과·논문흐름·실험프로토콜)
