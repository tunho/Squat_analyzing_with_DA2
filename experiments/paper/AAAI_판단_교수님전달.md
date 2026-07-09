# AAAI-27 목표에 대한 현황 보고 (교수님 전달용)

> 요지: **자산·실험은 견고하고 저널감입니다.** 다만 지난 3일 + 추가 세션 동안 이 자산에서
> AAAI급 general-ML 기여를 rigorous하게 탐색한 결과, **병목이 "데이터/노력"이 아니라
> "새 아이디어의 부재"** 임을 확인했습니다. 아래는 그 근거와, 그럼에도 AAAI를 노릴 경우
> 무엇이 필요한지입니다.

---

## 1. 한 줄 결론

markerless 무릎각 잔차보정 벤치마크(4추정기 × 4도메인 × GT)는 **저널 실적으로는 견고**합니다.
그러나 **현재 자산 그대로는 7/28 AAAI에 필요한 general-ML novelty가 나오지 않습니다.**
이유는 노력 부족이 아니라 구조적입니다(아래 3·4절).

## 2. 왜 venue가 문제인가

- AAAI(및 NeurIPS/ICML/CVPR)는 **새로운 방법·기전·이론**을 요구합니다.
- 우리가 가진 것은 **응용 벤치마크 + 임상 검증 + 정직한 실패 특성화**입니다 —
  이건 강력한 **저널 기여**이지, general-ML novelty가 아닙니다.
- "무릎각 보정"은 응용 주제라 AAAI main track과 **venue mismatch**입니다.
  일반화된 ML 기여가 별도로 있어야 AAAI 심사 테이블에 오릅니다.

## 3. 우리가 실제로 시도한 general-ML 각도 (재시도 방지용 기록)

"혹시 이건 해봤나?"에 대비해, 검증하고 죽은 방향을 정리합니다. **모두 재현된 결과입니다.**

| 시도한 방향 | 결과 / 왜 죽었나 |
|---|---|
| 선택적 보정 (거리·학습게이트·앙상블 불일치로 게이팅) | backfire를 예측하는 신호가 ~chance. 오라클 여지는 있으나 잡히지 않음 |
| 거리(support overlap)로 correctability 예측 | overlap↔이득 상관 ≈ +0.2 (거의 0) |
| 이종 추정기 융합(stacking) | within-domain +14%, **cross-domain 실패** + stacking 자체가 기존 기법 |
| TTA(temporal smoothness) | "smoothly wrong"(특히 MotionBERT) 때문에 오히려 backfire |
| few-shot affine(offset+scale) 보정 | do-no-harm만, 오라클 여지 회복 못 함. "공유shape×도메인scale" 가설 **기각** |
| depth-invariant 보정(z 제거) | 미미(6.10→6.00), 일관성 없음 |
| concept-shift 주장 | 도메인 98% 분리라 covariate shift와 **구분 불가**(confound) |
| 이종 disagreement = bias 민감 | 합성 d≤8만 지지, 실데이터 marginal. Agreement-on-the-Line/GDE와 정면 경쟁 |
| 이미지 기반 오차 예측 | "좌표 대신 이미지" = failure prediction(기존 장르), 새 아이디어 아님 |
| estimator×pose 오차 서명 / 방향성(히스테리시스) 오차 (금번 세션 추가 검증) | 오차 = 단조추세 + **도메인별 affine** + 불안정 비선형으로 분해됨. 전이되는 건 trivial한 추세뿐, 도메인 특이부는 이미 기각된 affine, backfire 재현. 히스테리시스는 ≈1.2°로 무시 수준 |

**공통 패턴:** 모든 방향이 (a) 기존 기법을 우리 데이터에 적용한 것이거나,
(b) 우리 데이터 자체에 의해 반증(covariate shift 지배 + 예측 불가한 backfire)됩니다.

## 4. 왜 구조적인가 (핵심)

우리의 유일 자산은 **"이질 추정기 4개의 (불)일치 × GT × 다도메인"** 입니다.
그런데 이 자산이 강하게 지원하는 질문의 종류가 정해져 있습니다 —
**disagreement / OOD-accuracy-prediction 계열**입니다.
그리고 그 계열이 정확히 **선행연구 밀집지대**입니다:

- Agreement-on-the-Line (Baek, NeurIPS'22)
- GDE (Jiang, ICLR'22)
- Disagreement Discrepancy (Rosenfeld & Garg, NeurIPS'23)
- Underspecification (D'Amour, JMLR'22)

즉 fresh하게 각도를 짜내도 **이미 점유된 영역**으로 반복 수렴합니다. 이게 방향 10개가
전부 막힌 우연이 아닌 이유입니다 — 자산이 그쪽으로만 질문을 밀어줍니다.

## 5. 병목은 데이터가 아니라 아이디어

- Phase A/B 벤치마크를 완성해도(현재 서버 진행 중) 그건 **더 좋은 벤치마크**일 뿐,
  AAAI novelty를 만들지 못합니다. 오히려 응용 벤치마크 성격을 강화합니다.
- 즉 **"실험을 더 돌리는 것"으로는 AAAI 문이 열리지 않습니다.**

## 6. 그럼에도 AAAI를 노린다면 — 필요한 것 (정직한 경로)

데이터를 더 파는 게 아니라, **밖에서 들어온 새 아이디어**가 있어야 합니다.
이 경우 우리 자산은 **훌륭한 검증 testbed**가 됩니다(하루 안에 검증 가능).

구체적으로 셋 중 하나가 필요합니다:
1. **교수님(또는 협업자)이 가진 구체적 novel 각도** — 만약 있으시면 그게 바로 빠진 조각입니다.
   저희가 4추정기×4도메인×GT로 즉시 검증하겠습니다.
2. novel 방향을 가진 연구자와의 **협업**.
3. 아이디어 배양 시간을 두고 **다음 사이클** 목표.

## 7. 확실한 대안 (병행 가능)

- **저널(IEEE TNSRE 또는 JBHI)**: 마감 없음, 우리 결과가 거의 그대로 실적이 됩니다.
  cross-domain 보정의 임상 배포 위험 + within-domain 신뢰성이라는
  actionable한 health-informatics 메시지로 프레이밍 가능합니다.

## 8. 교수님께 드리는 질문

- **혹시 염두에 두신 구체적인 novel ML 각도가 있으신가요?**
  있으시면 그게 저희가 못 찾은 빠진 조각이고, 즉시 데이터로 검증하겠습니다.
- 없으시다면, AAAI는 "현재 데이터를 더 파기"가 아니라 "새 아이디어/협업" 경로가 필요하며,
  그동안 **저널로 확실한 실적을 확보**하는 병행 전략을 제안드립니다.
