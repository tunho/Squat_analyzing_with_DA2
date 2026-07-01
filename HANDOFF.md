# 프로젝트 핸드오프 — 마커리스 3D 무릎각 추정·보정 / AAAI 탐색

> 새 Claude Code 세션용 컨텍스트. 여기에는 **자산·수행내용·발견·실패한 방향·정직한 결론**이 다 담겨 있음.
> 목적: (1) 무엇이 있는지 빠르게 파악 (2) 이미 검증해서 죽은 방향 **재시도 방지** (3) fresh 관점으로 새 각도 찾기.

---

## 0. 한 줄 요약 / 현재 정직한 상태

markerless pose estimator의 무릎각 오차를 **잔차보정**하는 벤치마크가 있음(견고, 저널감). 이 위에서 **AAAI급 novel 기여**를 3일간 rigorous하게 탐색했으나, **테스트한 ~10개 방향이 전부 weak/refuted/기존기법**. **정직한 결론: 우리 데이터+현재 아이디어로는 강한 AAAI 주장이 안 나옴. 병목은 데이터/방법이 아니라 "genuinely new idea 부재".** 저널(pose 벤치마크)은 확실한 실적. AAAI는 새 아이디어/협업/시간이 필요.

---

## 1. 목표

- **원래**: markerless 무릎각(굴곡/신전) 추정의 체계적 오차를 학습형 잔차보정으로 교정. 추정기-비종속 벤치마크.
- **당면 목표(사용자)**: **AAAI-27(마감 7/28) accept.** (실패 OK, 근데 accept이 목표. 저널은 마감 없어 후순위.)
- AAAI 벽: 무릎각 보정 = **응용 논문** → venue mismatch. 일반 ML novelty 필요.

## 2. 자산

### 데이터 (4 도메인, GT 무릎각 있음)
- **REHAB (ex6/REHAB24-6)**: 재활 스쿼트, ~9 피험자
- **FiT3D**: 피트니스 3D
- **SUMediPose (sumedipose_a3)**: Vicon gold GT, 26 피험자
- **AIHub (aihub_squat)**: 대규모 크로스핏 스쿼트 (NAS `/mnt/nas`, read-only)
- **원본 비디오/이미지 있음**: `dataset/raw/{aihub_squat,fit3d,REHAB24-6,sumedipose_a3}` (~5073 파일). **좌표만 안 쓰고 원본 픽셀 사용 가능.**
- NTU-RGBD (s018~)는 제외(Kinect GT 노이즈).

### 4개 추정기 (heterogeneous!)
MediaPipe(경량 CNN) · NLF · RTMW3D · MotionBERT(temporal transformer). 각 추정기가 4데이터셋에 돌아감. **이 "다른 아키텍처 4개 + GT + 다도메인" 조합이 유일 자산.**

### 추출 feature (좌표 기반)
`experiments/paper/features_{도메인}_{추정기}.csv` + `features_pooled4_{offstd|nlf|rtmw3d|motionbert}.csv`.
컬럼: frame_index, subject_id, view_type, mp_knee_angle, mp_hip_angle, k/a/s_x/y/z(관절좌표), *_vis(가시성), hip_depth_norm, thigh_len, shank_len, leg_ratio, angle_velocity/accel, lags 등. gt_angle 있음. **residual = gt_angle − mp_knee_angle.**
⚠️ feature 파일마다 컬럼셋 다름(nlf만 ex_type/take/domain/dataset 풍부, 나머지는 최소). **4추정기 프레임정렬 불가**(키 불일치, mediapipe는 다른 컨벤션). 3추정기(nlf/rtmw3d/motionbert)는 dedup keep=False로 정렬 가능.

### 코드/인프라
- `scripts/paper_baselines_loso.py` (잔차보정 LOSO, random_state=42), `gen_allcorr.py`(LODO/few-shot), `lodo_eval.py`, `agreement_stats.py`(ICC/Bland-Altman), corrector들(smoothnet/tcn/diffusion — **torch 시드 없음=재현성 이슈**).
- 결과: `experiments/paper/lodo_{추정기}/lodo_predictions.csv`(features+residual+predicted_residual+corrected_angle+dataset), `baselines_*`, `agreement_stats.csv`.
- venv: `.venv-paper`(sklearn/pandas), `.venv-nlf`/`.venv-mmpose`(torch/GPU).
- **compute**: 로컬 GPU RTX 4070Ti(16GB), 16코어. 학교 서버(junho@macs-station, GPU0,1만·CPU0-29만·GPU2,3=dongjun 금지)에서 전체 벤치마크(Phase A/B) 돌린 상태(완료 여부 서버 확인 필요).

## 3. 확립된 발견 (재현됨, seed42)

1. **잔차보정 within-domain 강력**: LOSO +14~70% (nlf AIHub 8.67→3.44 = 60%).
2. **cross-domain(LODO) backfire**: 보정이 오히려 악화 (예: mediapipe AIHub −11%, nlf→REHAB 4.71→7.32). residual-map 전이행렬 within 0.91 → cross 0.44.
3. **backfire가 per-instance 예측 불가**: 거리·학습게이트·앙상블불일치 다 ~chance(0.05~0.19).
4. **최고 단일 추정기가 도메인마다 뒤집힘**: nlf(REHAB/AIHub) ↔ motionbert(SUMe).
5. **학습 융합(stacking)**: within-domain 최고단일 +14%, **cross-domain 실패**(최고단일보다 나쁨).
6. **MotionBERT "smoothly wrong"**: 제일 매끄러운데 정확X. smoothness 가중 TTA는 backfire(9.7→24.9).
7. **오차는 굴곡각(자세)에 강하게 의존**, depth·visibility엔 거의 무관(Spearman~0.08).
8. **도메인 98% 분리** (RandomForest, 주로 깊이 z좌표 a_z/s_z/k_z). 심한 covariate shift.
9. **피험자 bias(±3.6°)는 관측특성으로 예측불가**(R²음수) = unmeasured latent.
10. **이종 추정기 disagreement > 동종**: pose 0.77 vs 0.15(단 confound), 합성 d≤8, folktables 5/5방향(약함).

## 4. 시도했다가 죽은 방향 (⚠️ 재시도 금지 — 이유 포함)

| 방향 | 결과 / 왜 죽음 |
|---|---|
| **선택적 보정** (거리·학습게이트·앙상블불일치 게이팅) | 신호가 backfire 예측 못 함(~chance). 오라클 헤드룸 있으나 못 잡음. |
| **거리(support overlap)로 correctability 예측** | overlap↔gain 상관 +0.2 (0에 가까움). REHAB backfire를 거리로 설명 못 함. |
| **이종 추정기 융합** | within +14%, cross-domain 실패. 게다가 stacking=기존. |
| **TTA (temporal smoothness)** | smoothly-wrong 때문에 backfire. |
| **few-shot affine(offset+scale) 보정** | do-no-harm만(backfire 막음), 오라클 헤드룸 못 회복. "공유shape×도메인scale" 가설 기각. |
| **depth-invariant 보정** (z 제거) | marginal(6.10→6.00), 일관성 없음. |
| **concept shift 주장** ("같은 자세 다른 오차") | **confound**: 도메인 98% 분리라 concept shift↔covariate shift 구분 불가. NN매칭 테스트 결과 가까운 매칭쌍은 residual 비슷(=covariate가 지배). 강하게 방어 안 됨. |
| **이종 disagreement = bias-sensitive (bias-var-diversity)** | 합성 d≤8 지지, d=16 붕괴. folktables **marginal**. **pose에선 진짜 동종 못 만듦(confound)**. pose 필요없는 순수 general-ML 주장이 됨 → moat 없고 효과 modest, Agreement-on-Line/GDE와 정면경쟁. |
| **이미지 기반 오차예측** | 실제 실행 안 함. 근데 "좌표 대신 이미지" = failure prediction(기존 장르), **새 아이디어 아님**. |

**공통 패턴**: within-domain은 잘 되나 cross-domain 전이가 근본적으로 안 됨(covariate shift 지배 + unmeasured bias). 그리고 우리가 제안한 모든 "방법"이 **기존 기법의 적용**이지 새 아이디어가 아님.

## 5. 관련 선행연구 (차별화 어려움 = novelty 벽)
Agreement-on-the-Line(Baek NeurIPS'22), GDE(Jiang ICLR'22), Disagreement Discrepancy(Rosenfeld&Garg NeurIPS'23), Underspecification(D'Amour JMLR'22), Wood 2023(diversity 분해), Cross-Model Disagreement 2026(LLM). 우리 발견 대부분이 이들에 인접/포함.

## 6. 정직한 전략 결론
- **AAAI 7/28 + accept + 우리 자산** = 동시 만족 불가(rigorous 확인). 병목 = **새 아이디어 부재**(데이터/방법 스왑으론 안 생김).
- **저널(pose 벤치마크 + 실패 특성화)** = 확실한 실적, 마감 없음.
- AAAI 현실 경로: (a) 시간+몰입(새 아이디어 emergence, 다음 사이클) (b) novel 방향 가진 사람과 협업.

## 7. Fresh 관점이 볼 만한 열린 지점 (검증 안 된 것)
- **원본 이미지/비디오** 아직 미사용. 좌표가 손실 표현이라 벽에 부딪혔을 수 있음 — 단 "이미지 오차예측"은 기존 장르라 **새 각도/기여가 필요**.
- **estimator 내부**(heatmap/confidence/latent) 미추출.
- 전신 스켈레톤·전체 시계열 궤적 미활용.
- **문제 자체를 바꾸기**: 무릎각 응용을 벗어나, 우리 "다추정기×다도메인×GT" 셋업을 general ML 질문의 testbed로 쓰되 **진짜 novel 질문**을 찾아야(아직 못 찾음).

## 8. 실무 노트
- 경로: `/home/lee/projects/exe_est`. 저장소 git 있음.
- LODO 데이터로 빠른 분석: `experiments/paper/lodo_nlf/lodo_predictions.csv` (features+residual+dataset 다 있음, 제일 편함).
- 재현성: sklearn(extratrees/mlp) seed42 OK, **torch 3종 시드 없음** → AAAI용은 전시드고정+멀티시드 필요.
- NAS `/mnt/nas` read-only, JHL 폴더만 스코프, 쓰기/삭제 금지.
- 서버 실험(Phase A/B) 상태는 서버 Claude(junho@macs-station)에게 확인.

---

## 새 세션 시작 프롬프트 (복붙용)

```
이 저장소(/home/lee/projects/exe_est)의 HANDOFF.md를 먼저 읽어줘. markerless 무릎각 추정·잔차보정 프로젝트고, AAAI-27 novel 기여를 찾다가 ~10개 방향이 다 weak/refuted/기존기법으로 판명났어(HANDOFF.md 4절에 죽은 방향+이유 정리됨 — 재시도 금지).

내가 원하는 것: [여기에 목표를 적어 — 예: "fresh 관점으로 우리 자산(4추정기×4도메인×GT+원본비디오)에서 genuinely novel한 AAAI 각도를 찾아줘. 단 HANDOFF 4절의 죽은 방향은 빼고, 기존기법 적용이 아닌 새 아이디어여야 해." 또는 "저널용 pose 벤치마크 논문을 정리하자" 등]

정직하게: novel 아이디어가 없으면 없다고 말하고, 억지 방향 제안하지 마. 데이터로 빠르게 검증 가능한 것부터.
```
