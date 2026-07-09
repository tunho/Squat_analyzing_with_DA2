# Cross-domain offset 추정: 시도와 결과

> 목적: cross-domain에서 잔차보정이 backfire하는 근본 원인 = **도메인별 residual offset을 관측특성으로 식별할 수 있는가?** 를 정리.
> 결론(먼저): **불가.** offset은 실재하나 depth(z) 포함 어떤 관측특성으로도 신뢰성 있게 추정되지 않음 → cross-domain 보정이 원리적으로 전이 불가. within-domain은 LOSO가 offset을 암묵 흡수해 성립.
> 단위: 각도 MAE(°), NLF 추정기, LODO(leave-one-domain-out), ExtraTrees residual, seed42.

---

## 0. 배경 사실 (재확인)
- **within-domain 보정은 강력** (LOSO), **cross-domain은 backfire** (corrected MAE > raw MAE), 특히 REHAB.
- 도메인 offset(평균 residual): AIHub +6.77, FiT3D +5.98, SUMediPose +7.71, **REHAB −0.14**. REHAB만 offset이 0 근처 → 다른 도메인으로 학습한 양(+)의 offset을 REHAB에 적용하면 과보정 → backfire.

## 1. fps/프레임 밀도가 원인인가? → **기각 (통제실험 2종)**
프레임당 velocity는 실제 속도가 아니라 fps 함수임을 확인:

| domain | rep당 프레임 | 프레임당 \|velocity\| |
|---|---|---|
| FiT3D | 508 | 2.9 |
| REHAB | 232 | 0.9 |
| AIHub | 11 | 13.3 |
| SUMediPose | 11 | 22.0 |

그러나 fps는 cross-domain backfire의 원인이 아님:
- **fps-독립(static) 피처만으로 LODO**: REHAB raw 4.71 → corrected **6.99 (BACKFIRE 유지)** (full-feature 7.10과 거의 동일).
- **프레임 밀도 균등화 후 LODO** (모든 도메인 ~11 frames/rep로 subsample): REHAB raw 5.16 → **6.02 (BACKFIRE 유지)**.
→ backfire는 fps/샘플링이 아니라 spatial shift.

## 2. 진짜 shift는 depth(z) 영역 차이 (fps 무관)
깊이 좌표 `a_z` 중앙값: AIHub +0.36, FiT3D +0.15, SUMediPose +0.13, **REHAB −0.03 (p10 −0.20, 음수)**. 무릎각 범위는 도메인 간 유사(p10 ~75–86°)한데 **REHAB만 depth 영역이 딴 세상** = 카메라 셋업/좌표계 차이. (이전 세션 finding 8 "도메인 98% 분리 = z좌표 주도"와 일치.)

## 3. 그러면 z로 offset을 추정할 수 있나? → **불가 (3각도)**
**(a) within-domain: z↔residual 상관 약함/비일관**, 게다가 자세(gt_angle)와 교락.

| domain | Spearman(resid, a_z) | Spearman(resid, gt_angle) |
|---|---|---|
| AIHub | −0.34 | −0.36 |
| FiT3D | −0.34 | −0.46 |
| REHAB | −0.16 | −0.27 |
| SUMediPose | −0.09 | −0.37 |

**(b) 도메인 간 offset↔z: 겉보기 corr 0.72지만 n=4, REHAB 단일 outlier가 주도** → 통계적으로 무의미.

**(c) z-only 피처로 cross-domain 보정 → 가장 다른 REHAB에서 폭망:**

| held-out | raw | z-only corrected | R² | 판정 |
|---|---|---|---|---|
| **REHAB** | 4.71 | **7.87** | **−1.45** | BACKFIRE |
| FiT3D | 6.68 | 4.05 | +0.09 | ok |
| SUMediPose | 9.39 | 6.87 | −0.07 | ok(offset 제거 수준) |
| AIHub | 8.68 | 6.61 | +0.22 | ok |

z-only는 3개 도메인에서 대략적 offset만 제거할 뿐, 정작 offset이 다른 REHAB에서 R²=−1.45로 최악.

## 4. 이전 세션의 offset 추정 시도 (HANDOFF §4, 전부 실패)
- **거리/support overlap → correctability**: 상관 +0.2 (≈0). REHAB backfire 설명 못 함.
- **학습 게이트 / 앙상블 불일치 게이팅**: backfire 예측 ~chance (0.05~0.19).
- **few-shot affine (offset+scale)**: do-no-harm만, 오라클 헤드룸 회복 실패.
- **depth-invariant 보정 (z 제거)**: marginal(6.10→6.00), 비일관.
- **피험자 bias를 관측특성으로 예측**: R² 음수 (unmeasured latent, finding 9).
- **공유/추정기고유 오차분해 + 전이 (이번 세션)**: 공유항 약함(추정기간 corr 0.07~0.37), cross-domain 전이 R² +0.13/+0.27/−0.07/−0.43, 적용 시 4/4 backfire.

## 5. 종합 결론
- **z는 도메인 마커이지 residual 예측자가 아님.** "이건 REHAB이다"는 알려주나 "여기 오차가 +몇 도다"는 못 알려줌.
- 도메인 offset은 실재하나 **관측 피처(depth 포함)로 식별 불가** → 카메라 intrinsic·피험자 population·추정기×도메인 상호작용 등 **unmeasured latent**에 묶임.
- 따라서 **관측→offset 보정이 cross-domain 전이 불가** = 모든 cross-domain 방향이 죽은 근본 이유.
- **함의**: within-domain은 LOSO가 offset을 암묵 흡수해 성립. cross-domain은 원리적으로 불가. 저널에서는 이를 **정밀한 실패 특성화**("도메인 오차는 실재하나 depth 포함 관측특성으로 식별 불가, fps 아티팩트 아님(통제됨)")로 사용.
