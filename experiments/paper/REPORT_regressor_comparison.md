# 회귀기 비교 — 스쿼트 무릎각 잔차 보정

## 실험 설정
- 데이터: AIHub 스쿼트 (44명, 109,934 프레임, 8화각)
- 검증: 피험자단위 Leave-One-Subject-Out (LOSO)
- 특징: `enhanced_safe` 39개 (하체 좌표·속도·가시성·분절안정성·화각)
- 목표: `residual = GT각 − MediaPipe각`, 보정각 = MediaPipe각 + 예측잔차
- 전처리: StandardScaler (전 모델 동일), **회귀기만 교체**
- 지표: 전체(mean) 및 깊은굴곡(<110°) MAE, LOSO 피험자평균

## 비교 모델
| 모델 | 설정 |
|---|---|
| Linear (Ridge) | alpha=1.0 (단순 선형 baseline) |
| HistGBM | max_iter=400, lr=0.05 |
| LightGBM | n_estimators=400, num_leaves=63, lr=0.05 |
| XGBoost | n_estimators=400, max_depth=8, lr=0.05 |
| ExtraTrees (현행) | n_estimators=200, max_depth=15 |

## 결과 (개선율 높은 순)

| 회귀기 | 전체 raw→cor | 전체 개선 | 깊은굴곡 raw→cor | 깊은굴곡 개선 |
|---|---:|---:|---:|---:|
| **ExtraTrees (현행)** | 17.95→10.22 | **43.1%** | 25.18→11.39 | **54.8%** |
| XGBoost | 17.95→10.27 | 42.8% | 25.18→11.58 | 54.0% |
| LightGBM | 17.95→10.36 | 42.3% | 25.18→11.62 | 53.8% |
| HistGBM | 17.95→10.47 | 41.6% | 25.18→11.73 | 53.4% |
| Linear (Ridge) | 17.95→15.26 | 15.0% | 25.18→17.07 | 32.2% |

## 결론
- 비선형 앙상블(트리/부스팅)이 단순 선형(15.0%)보다 압도적으로 우수.
- 앙상블 4종은 41.6~43.1%로 유사하며, **ExtraTrees(현행)가 전체·깊은굴곡 모두 최고**

> 비고: RandomForest는 동일 조건에서 연산시간이 오래걸려서 제거 했습니다. 
