# 추정기 (Pose Estimator)

| 모델 | 발표 | 논문 링크 |
|---|---|---|
| MediaPipe BlazePose | 2020 | https://arxiv.org/abs/2006.10204 |
| NLF | NeurIPS 2024 | https://papers.nips.cc/paper_files/paper/2024/file/fd23a1f3bc89e042d70960b466dc20e8-Paper-Conference.pdf |
| RTMW3D | 2024 | https://arxiv.org/abs/2407.08634 |
| MotionBERT | ICCV 2023 | https://arxiv.org/abs/2210.06551 |
| WHAM | CVPR 2024 | https://arxiv.org/abs/2312.07531 |

# 보정기 (Corrector)  — 확산모델(DRPose 등)은 교수님 지시로 보류, 아래만 진행

| 방법 | 발표 | 논문 링크 |
|---|---|---|
| SavGol | 1964 | (scipy, 고전 평활) |
| Kalman | 1960 | (scipy, 고전 평활) |
| SmoothNet | ECCV 2022 | https://arxiv.org/abs/2112.13715 |
| TCN | 2018 | https://arxiv.org/abs/1803.01271 |
| MLP | 1986 | https://doi.org/10.1038/323533a0 |
| ExtraTrees(제안) | 2006 | https://doi.org/10.1007/s10994-006-6226-1 |

# 보류 (나중에) — 확산 기반 보정기
DRPose(2024) https://arxiv.org/abs/2401.04921 · D3PRefiner(2024, 코드없음) · DPoser-X(2025)
※ 교수님 지시: 확산모델은 일단 나중에, 먼저 제안한 것(위)만.
