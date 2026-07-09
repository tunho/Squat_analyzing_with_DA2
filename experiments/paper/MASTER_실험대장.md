# 마스터 실험 대장 — 전체 (로컬 세션 + 서버 통합)

> markerless 무릎각 잔차보정 프로젝트의 **모든 실험**(성공·실패·ablation)을 한 파일에 통합. 재현·기록·교수님 전달·논문용.
> 통합 소스: 서버 `실험종합대장.md`(A~F) + 서버 `outB_*`(Phase B 확정) + 로컬 `clinical_utility/`(임상·기전) + `HANDOFF.md`(§4 죽은 방향). 갱신 2026-07-03.
> 출처표기: **[서버]** = macs-station 산출 / **[로컬]** = 이번 세션 직접 실행 / **[기록]** = 이전 세션 기록.
> 단위: 각도 MAE(°). 제안 보정 = ExtraTrees 잔차회귀(target=gt−mp, seed42). 향상%는 raw 대비, 음수=backfire.

---

## Part 0. 자산 · 설정

- **추정기 4종**(heterogeneous): MediaPipe(경량 CNN) · NLF · RTMW3D · MotionBERT(temporal transformer).
- **도메인 4종**(GT 무릎각): REHAB24-6(재활 스쿼트 9명) · FiT3D(피트니스 3명) · SUMediPose(Vicon gold 26명) · AIHub(크로스핏 44~46명).
- **평가**: Phase A = LOSO(leave-one-subject-out, 도메인 내) / Phase B = LODO+few-shot(leave-one-dataset-out, 도메인 일반화).
- **보정기 9종**: SavGol·Kalman(평활) / ExtraTrees·MLP(sklearn)·mlptorch(GPU) 잔차 / SmoothNet·TCN·Diffusion(CARD) 시계열 / ExcelFormer·TabM(tabular DL).

### Part 0b. 방법 검증 — 논문 원전 대비 (2026-07-08 확인)
**추정기 = 전부 공식 모델/가중치로 실제 추론:**
- MediaPipe: 공식 MediaPipe Pose(BlazePose), `src/squat/models/pose_landmarker.py`.
- **NLF** (NeurIPS'24, arXiv 2407.07532): 공식 **isarandi/nlf TorchScript 가중치**를 `torch.jit.load`+`predict_joints3d`로 추론 (`paper_extract_nlf.py`, `.venv-nlf`).
- **RTMW3D** (arXiv 2407.08634): 공식 **mmpose 1.3.2** (`.venv-mmpose`), `sumedipose_est.py`/`paper_extract_fit3d_est.py`.
- **MotionBERT** (ICCV'23, arXiv 2210.06551): 공식 체크포인트, `coco17_to_h36m17`+`lift(normalize_screen(...))` (`paper_extract_motionbert.py`, `--mb-ckpt`).

**보정기 = 공식 패키지 또는 논문 아키텍처 충실 구현:**
- ExcelFormer(KDD'24, 2301.02819): 공식 **pytorch-frame** ExcelFormer. TabM(ICLR'25, 2410.24210): 공식 **tabm** pip.
- SmoothNet(ECCV'22, 2112.13715): 시간축 FC residual망(Linear→ResBlock+LayerNorm→decoder) 구조 재현.
- TCN(**Bai et al. 2018**, 1803.01271; 개념 최초는 Lea 2016, 1608.08242): Chomp1d+dilated Conv1d+residual TemporalBlock 재현.
- Diffusion/CARD(NeurIPS'22, 2206.07275): f_φ(x)앵커 MeanEstimator+Denoiser+cosine β+T=100 재현.
- ExtraTrees·MLP=sklearn, SavGol=scipy, Kalman=표준.

**⚠️ 논문 Methods에 명시할 nuance**: SmoothNet·TCN·CARD는 **아키텍처는 원전 그대로**지만 **입력을 우리 과제(무릎각 시계열·잔차)에 적응**해 baseline으로 사용 (원전은 pose좌표 시퀀스/일반회귀). "우리는 X의 아키텍처를 잔차 보정에 적용" 식으로 기술.

---

## Part 1. Phase A — within-domain 보정 (LOSO) [서버]

### A1. ExtraTrees 잔차보정 16칸 (raw°→corr°, +향상%)
| 도메인 | MediaPipe | NLF | RTMW3D | MotionBERT |
|---|---|---|---|---|
| REHAB | 12.3→8.2 (+34%) | 4.8→4.1 (+15%) | 14.8→5.9 (+60%) | 10.4→6.3 (+39%) |
| FiT3D | 20.2→13.4 (+34%) | 6.7→4.7 (+30%) | 11.1→10.2 (+8%) | 9.6→6.2 (+35%) |
| SUMe | 12.6→7.0 (+44%) | 9.2→5.4 (+42%) | 12.8→5.7 (+55%) | 6.7→5.4 (+20%) |
| AIHub | 14.3→7.0 (+51%) | 8.8→3.8 (+57%) | 18.9→5.4 (+71%) | 28.3→4.9 (+83%) |

→ **16칸 전부 +8~83%.** raw 클수록 이득 큼.

### A2. 보정기 8종 벤치마크 (16칸 LOSO 향상%) — + ExcelFormer(KDD 2024) 추가
- **ExtraTrees만 파국 0** (전 칸 양수). 트리는 학습범위 밖 외삽을 안 해 붕괴 없음.
- 실패모드: **평활기(SavGol/Kalman) 거의 무효**(+0~2%), **MLP 소규모 −29~−35%**, **Diffusion 저오차서 −44~−51%**, SmoothNet/TCN 중간.
- 예: ex6_NLF — SavGol +0.7 / MLP −29.5 / Diffusion −51.4 / **ExtraTrees +13.2**.
- **[2026-07-05 추가] ExcelFormer (KDD 2024 게재·코어 arXiv 2023, tabular DL baseline, pytorch-frame, 전체데이터·서브샘플없음, epochs=40)**: 평균 **+34.3%**(파국 1) vs ExtraTrees +39.2%. **14/16칸에서 ExtraTrees ≥ ExcelFormer** — 소규모(fit3d/ex6) 트리 확연 우세, **대규모 AIHub만 동등/역전**(aihub_rtmw3d +72.7 vs +71, aihub_motionbert +83.9 vs +83). 파국은 fit3d_rtmw3d −2.2% 하나(경미). (caveat: 미튜닝, "2024 최신 SOTA"는 과장 — 코어 2023.) 결과: `excelformer_out/summary_excelformer.csv`.
- **[2026-07-06 추가] TabM (ICLR 2025, arXiv 2410, 가장 최신 tabular DL, `tabm` pip, k=16·GPU 1개 사유, epochs=40)**: within 16칸 평균(±100클립) **+25.5%, 파국 4칸** vs ExtraTrees +39.2%(파국1)·ExcelFormer +34.3%(파국1). **대규모(AIHub)에선 트리를 이김**(+54.9/+56.1/+76.0/+84.9 ≥ 트리 +51/+57/+71/+83), **소규모(FiT3D 3명)에선 붕괴**(`fit3d_rtmw3d −251.7%`, 11°→38°). LODO backfire·few-shot 회복은 트리·ExcelFormer와 동일(단 FiT3D는 few-shot도 악화). → **"진짜 2025 최신 딥러닝도 트리 못 넘고, 오히려 소규모에서 가장 극적으로 붕괴"** — "데이터 적은 임상엔 트리가 안전" 논지 강화. 결과: `tabm_out/summary_tabm.csv`. 시드 재현성 인프라(`seed_util.py`·`run_multiseed_within.py`) 별도 구축.

### A3. 회귀기 비교 (AIHub) [서버]
비선형 앙상블 41.6~43.1% ≫ 선형 15%. **ExtraTrees 최고(43.1%, 깊은굴곡 54.8%)** > HistGBM/LightGBM/XGB.

### A4. pooled LOSO (미관측 subject)
**ExtraTrees**: MediaPipe +42.9 / NLF +48.9 / RTMW3D +60.8 / MotionBERT +70.0% (평균 +55.7). 대규모선 MLP>ExtraTrees(규모효과).
- **[2026-07-07 추가] pooled 보정기 비교 (전부 84-fold LOSO + seed42 통일, ①③④와 동일 프로토콜)**: 평균 **ExtraTrees +55.7 > TabM +51.3 > ExcelFormer +46.2**. 칸별 ExtraTrees 2승(NLF·MotionBERT)·TabM 2승(MediaPipe·RTMW3D)·ExcelFormer 0승. within은 ExcelFormer>TabM이나 **pooled(데이터↑)선 TabM>ExcelFormer 역전**(규모효과), 그래도 트리 못 넘음. 실행: 로컬 TabM(4070Ti)+서버 ExcelFormer(V100×2 raypool, GPU0,1·CPU0-29). 결과 `tabm_out/pooled_*`·`excelformer_out/pooled_*`.

### A5. 임상 일치도 (ICC/Bland-Altman/CCC, 16칸+pooled)
bias→~0, **ICC<0.9 9칸 → 보정 후 0칸**(평균 0.972), Wilcoxon 전부 유의.

### A6. 임상 지표 (Depth/ROM MAE, 깊이분류)
깊이분류 99.6%+, **NLF 깊이오차 4.65°**(임상 <5°).

### A7. 잔차 vs 직접 회귀 ablation
평균 동등하나 **잔차가 저오차 도메인 파국 회피**(fit3d_NLF 잔차 +27 vs 직접 −9.5) = 안전성 hook.

**Part 1 결론:** 잔차보정은 추정기·도메인 비종속으로 작동(+8~83%). 단일 최우수 보정기 없음, **ExtraTrees만 전 조건 견고.** → 저널 본체.

---

## Part 2. Phase B — cross-domain 일반화 (LODO + few-shot) [서버 outB]

### B1. LODO 향상% (held-out 도메인 통째 미관측, ExtraTrees)
| 추정기 | AIHub | SUMe | REHAB | FiT3D |
|---|---|---|---|---|
| MediaPipe | **−13** | +24 | +9 | +22 |
| NLF | +29 | +28 | **−49** | +31 |
| RTMW3D | +6 | +33 | +29 | +3 |
| MotionBERT | **−18** | **−42** | **−7** | **−22** |

→ **추정기 의존·비일관.** RTMW3D 전이 견고, MotionBERT 전 도메인 backfire, NLF→REHAB −49%.

### B2. few-shot 회복 (target subject k명 추가, pooled 향상%)
| 셀 | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| NLF→REHAB | −49.1 | −34.0 | −23.4 | −14.0 |
| MotionBERT→AIHub | −18.3 | +24.8 | +58.2 | **+64.0** |
| MotionBERT→SUMe | −41.6 | −10.2 | −1.3 | +5.2 |
| MediaPipe→AIHub | −12.8 | +14.0 | +24.4 | +27.7 |

→ **target 소량 감독으로 backfire 회복**(대부분 k=1~3). 근본한계 아닌 분포·offset 정렬 문제. = 임상 배포 가이드("몇 명 필요한가").

### B3. 신경망 보정기 시드고정 (mlptorch, GPU) [서버]
sklearn MLP 동등설계(256,128/Adam/300ep) GPU 재현판. `outB_*/lodo_mlptorch.csv`, `fewshot_mlptorch.csv` 전 추정기 완비 → **제출 블로커 해소.**

### B4. Phase B 보정기 비교 (LODO 향상% 4도메인 평균, 괄호=backfire 셀수/4)
| 추정기 | ExtraTrees | MLPtorch | SmoothNet | TCN | Diffusion |
|---|---|---|---|---|---|
| MediaPipe | **+10.5 (1)** | −3.4 (2) | +9.7 (1) | +13.3 (1) | +0.8 (2) |
| NLF | **+9.7 (1)** | −23.7 (2) | −11.0 (1) | −13.3 (1) | −20.1 (3) |
| RTMW3D | **+17.8 (0)** | −4.1 (2) | −0.4 (2) | −1.7 (3) | +3.4 (2) |
| MotionBERT | −22.2 (4) | −30.5 (4) | **−3.8 (2)** | −4.0 (3) | −21.0 (4) |

- **ExtraTrees가 cross-domain에서도 최소 파국**(RTMW3D 0 backfire, 3추정기서 최고). 학습형 중 유일하게 견고.
- **REHAB backfire는 전 보정기 공통**(NLF→REHAB: extratrees −49 / mlptorch −85 / smoothnet −49 / tcn −59 / diffusion −93). **MotionBERT도 전 보정기 backfire.** → 보정기 트릭 문제 아니라 **도메인 문제** 확증.

---

## Part 3. 임상 유용성 (within-domain 보정의 임상 번역) [로컬]

### C1. 임상 허용오차(±5°) 이내 비율 (16칸 평균)
| 지표 | 전 → 후 |
|---|---|
| frame ±5° | 33.4% → **52.7%** |
| frame 심굴곡 ±5° | 20.7% → **45.6%** |
| rep 최대굴곡 MAE | 15.3° → **7.2°** |
| rep 최대굴곡 ±5° | 20.8% → **47.9%** |

### C2. 베이스라인 비교 (frame ±5°, FiT3D 제외 평균)
raw 34.8 / **SavGol 29.7(⬇)** / 전역offset 34.0 / **학습형 56.8** → 스무딩·단순offset 임상치 못 넘고 학습형만 넘김(~2배).

### C3. 통계 [로컬 stats_batch]
- **Bland-Altman(Vicon gold=SUMe)**: bias **−13.0°→0.4°**(체계오차 제거), REHAB −4.9°→0.4°. **LoA(산포) 유지**(한계).
- **CCC**: 보정 후 대부분 **>0.92**(평균 0.972).
- **유의성**: per-subject Wilcoxon, powered 3도메인 전 추정기 **p ≤ 0.008**. 부트스트랩 CI 대부분 0 배제.

### C4. 배포 현실형 (GT 없이 추정신호로 rep 검출)
최대굴곡 MAE 16.3°→**6.1°**, ±5° 22.9→53.5%(GT-이상형과 동등). **rep 검출기 recall 1.00/precision 0.83.** 보정이 rep 검출 안정화(RTMW3D raw 660→보정 454, GT≈452).

### C5. 보정기별 임상 depth MAE 향상% [서버 out_*/clinical_*]
| 추정기 | ExtraTrees | MLP | SavGol | Kalman | SmoothNet | TCN | Diffusion |
|---|---|---|---|---|---|---|---|
| MediaPipe | +46 | +47 | **−49** | **−112** | +15 | +14 | +53 |
| NLF | +57 | +52 | −0 | +16 | +49 | +50 | +50 |
| RTMW3D | +58 | +58 | −0 | −5 | +27 | +27 | +60 |
| MotionBERT | +68 | +66 | −0 | −2 | +27 | +23 | +67 |

→ **학습형(ExtraTrees/MLP/Diffusion)만 임상 depth 정확도 도달(+46~68%, MAE ~4.7~6.8°).** **스무딩(SavGol/Kalman)은 임상 depth에서도 실패**(0~−112%). C2(frame ±5°)를 임상 측정단위(깊이)에서도 확증. (scope=전체, AIHub 지배 n=506/527.)

---

## Part 4. 기전 · 오차구조 · 실패 특성화 [서버 D + 로컬]

### D1~D4. 오차구조 분해 [서버]
- **D1** estimator×pose 오차서명: Pearson 곡선상관 mp +0.93…mb +0.22 → "도메인 불변 서명"처럼 보였으나
- **D2** confound: Pearson은 affine 불변 → 곡선이 affine 관계일 뿐. raw cross-domain vs 오라클-affine: rtmw3d 13.9/3.5, mb 14.3/3.8 → **D1 붕괴**(=기존 C6 affine).
- **D3** detrend 후 잔여구조 전이: corr mp +0.79, nlf +0.14, mb +0.17 (비일관).
- **D4** 방향성(히스테리시스): |gap|≈1.2°, 부호일관 62%(≈chance) → 무시.

### D5. subject bias ↔ 실제 인구통계 (SUMe n=26) [서버]
BMI/체중 약상관(nlf weight r=+0.39, rtmw3d bmi r=+0.42), skintone·age·height null. **다중비교(48검정) 보정 미통과, R²≈16%** → 대부분 미설명.

### 오차 결정요인 [기록]
오차는 **자세(굴곡각)에 강하게 의존**, depth·가시성엔 무관(Spearman ~0.08). → within-domain 학습 성립 근거.

### fps confound 기각 (통제 2종) [로컬]
프레임당 velocity=fps 함수(rep당 프레임 FiT3D 508→\|vel\|2.9 vs SUMe 11→22.0). 그러나 ①static-only LODO(REHAB 4.71→6.99) ②프레임밀도 균등화 LODO(5.16→6.02) **둘 다 backfire 유지** → fps 아티팩트 아님.

### cross-domain 원인 = depth covariate shift [로컬]
깊이 a_z 중앙값 REHAB **−0.03** vs 타도메인 +0.13~0.36(카메라 차이). 도메인 98% 분리(z 주도, 무릎각 범위는 유사).

### offset 관측특성으로 식별 불가 [로컬 offset_estimation_findings.md]
z-only cross-domain 보정 REHAB **R²=−1.45**(폭망). 도메인간 corr(mean_az, offset)=0.72지만 n=4 REHAB outlier. → **z는 도메인 마커지 residual 예측자 아님.** offset은 unmeasured latent.

**Part 4 결론:** within-domain은 LOSO가 offset 암묵 흡수해 성립, **cross-domain은 offset 관측불가라 원리적 전이 불가**, few-shot 소량 라벨로 회복.

---

## Part 5. AAAI novelty 탐색 — 전부 실패 (재시도 금지) [기록+로컬]

| # | 가설/시도 | 결과 | 판정 |
|---|---|---|---|
| C1 | 선택적 보정(거리·게이트·불일치로 backfire 사전차단) | 세 신호 backfire 예측 ~chance(0.05~0.19) | 기각 |
| C2 | correctability=support overlap | overlap↔이득 상관 +0.2(≈0) | 기각 |
| C3 | residual-map 전이성 | within 0.91→cross 0.44, 예측/활용 불가 | 관찰만 |
| C4 | 이종 추정기 융합(stacking) | within +14%, cross 실패 + stacking=기존 | 기각 |
| C5 | TTA(temporal smoothness) | MotionBERT smoothly-wrong로 backfire 9.7→24.9° | 기각 |
| C6 | few-shot affine(offset+scale) | do-no-harm만, 오라클 회복 실패 | 기각 |
| C7 | depth-invariant 보정(z 제거) | 6.10→6.00° 미미·비일관 | 기각 |
| C8 | concept-shift 주장 | 도메인 98% 분리라 covariate와 구분 불가(confound) | 기각 |
| C9 | 이종 disagreement=bias 민감 | 합성 d≤8만, folktables marginal, pose confound. AGL/GDE와 경쟁 | 약함 |
| C10 | subject bias 예측 | bias ±3.6° R² 음수(예측불가) | 기각 |
| C11 | 오차 결정요인 | 굴곡각 의존, depth/vis 무관(0.08) | 관찰(논문용) |
| C12 | 이미지 기반 오차예측 | 좌표→이미지=failure prediction(기존 장르) | 개념적 기각 |
| C13 | **공유/추정기고유 오차분해+전이** [로컬] | 공유항 약함(추정기간 corr 0.07~0.37), 전이 R² +0.13/+0.27/−0.07/−0.43, 적용 4/4 backfire | 기각 |
| C14 | **스쿼트 폼 분류** [로컬] | REHAB correctness 라벨 내장하나, gold GT로도 Ex6 LOSO AUC≈0.52 = chance. 운동학 무관 | 기각 |

**공통 패턴:** 모든 방향이 (a) 기존기법의 우리 데이터 적용 또는 (b) 우리 데이터에 반증(covariate shift 지배 + 예측불가 backfire). **병목 = 새 아이디어 부재.**

---

## Part 6. 정직한 한계 (Limitations)

- **FiT3D 저파워**(3 subject) — 유일하게 backfire·비유의 다발.
- **MotionBERT "smoothly wrong"** — ROM 개선 미미, cross-domain 전 도메인 backfire.
- **LoA(무작위 산포) 미개선** — 보정은 체계 bias 제거지 정밀도 향상 아님.
- **concept vs covariate shift 구분 불가**(도메인 98% 분리 confound).
- **단관절(무릎) case study** — AIHub는 각도 GT만.
- **backfire 물리원인 미규명** — depth covariate shift까지, BMI/체중 약연관은 다중비교 후 소멸.

---

## Part 7. 상태 · 제출 게이트

- **무거운 실험·제출 블로커 전부 완료**: Phase B 매트릭스(E2) + 신경망 시드고정(E3) = 서버 / 임상·fps·offset = 로컬.
- **남은 것 = 정리·집필.** (선택: 신경망 보정기 멀티시드 mean±std = rigor용, 블로커 아님.)
- **타깃 저널**: J-BHI(서버 README) 또는 IEEE T-NSRE.
- **AAAI**: 접음(novelty 14방향 전부 죽음). 밖에서 새 아이디어 오면 본 자산이 하루 검증 testbed.

---

## Part 8. 파일 맵 · 재현

| 범주 | 위치 |
|---|---|
| Phase A LOSO 상세 | 서버 `out_<est>/method_comparison.csv`, `predictions_*.csv` / 로컬 `RESULTS_ALL_KR.md`, `loso_<dom>_<est>/` |
| Phase B LODO·few-shot | 서버 `outB_<est>/lodo_*.csv`, `fewshot_*.csv` (로컬 `_server_bundle/`에 백업) |
| 임상 유용성·기전 | 로컬 `experiments/paper/clinical_utility/` (metrics CSV·그림 6종, `offset_estimation_findings.md`) |
| 코드 | `gen_allcorr.py`(Phase B), `torch_mlp_corrector.py`(GPU MLP), `paper_baselines_loso.py`(Phase A), `paper_train_loso.py`, `paper_common.py` |
| 이력·기록 | `HANDOFF.md`(§4 죽은 방향), 서버 `실험종합대장.md`·`교수님_분석노트.md` |
