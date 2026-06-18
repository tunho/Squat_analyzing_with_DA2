# D. 카메라 기하 분석 실행 계획 (에이전트용)

목표: **카메라 기하(거리·높이·각도)가 MediaPipe 무릎각 오차를 좌우함을 정량 증명**하고, cross-dataset 전이(AIHub→REHAB −20% 등)를 카메라기하로 설명.
상태: [ ]대기 [~]진행 [x]완료

---

## 입력 데이터 (카메라기하 복원 소스)
| 데이터셋 | GT 2D | GT 3D | 제공 calib | 복원법 | 단위 |
|---|---|---|---|---|---|
| AIHub | local_keypoints CSV (cam별, 9648) | 3d_points.csv (mm) | ❌ | **DLT(2D-3D)** | mm |
| REHAB | 2d_joints/.npy (cam c17/c18, 26관절) | 3d_joints/.npy (26관절) | ❌ | **DLT** | ? 확인 |
| SUMediPose | internal_data JSON `xy`(63) | JSON `xyz` | ✅ extrinsics+intrinsics | **DLT + 제공calib 검증** | ? 확인 |
| FiT3D | ❌ | joints3d_25 | ✅ camera_parameters(R,T,K) | **제공 calib 직접** | m |

주의:
- **관절 대응**: 2D·3D가 같은 관절 집합·순서여야 함(셋별 확인 필요).
- **프레임 대응**: 2D·3D 프레임 정합(인덱스 일치) 확인.
- **단위 통일**: 거리 cross-dataset 비교 시 **피험자 키로 정규화**(거리/키) — mm/m 혼재 + 체형차 제거.
- **DLT 조건**: 비공면(non-coplanar) 점 필요 → 여러 프레임(다양 자세) 풀링이면 OK. 프레임 서브샘플(예: 200점)로 충분.

---

## 단계

### D0. [ ] 카메라기하 추출기 `scripts/camera_geom.py`
- DLT셋(AIHub·REHAB·SUMediPose): (subj,cam)별 → 2D-3D 대응 N프레임 풀링 → **DLT로 3x4 투영행렬 P** 추정 → 분해 K,R,T → **카메라중심 C = -Rᵀ·T**(GT 3D world 좌표계).
  - cv2: `cv2.calibrateCamera` 또는 수동 DLT(SVD) + RQ분해. (intrinsics 모르니 full calib)
- FiT3D: camera_parameters R,T → C = -Rᵀ·T 직접.
- 각 (dataset,subj,cam): **거리**=‖C−피험자중심(GT3D 평균)‖, **높이**=C 수직성분(피험자 기준), **elevation/azimuth** 각도.
- 산출: `outputs/paper/camera_geom/camera_positions.csv` [dataset,subject,camera,dist,dist_norm(÷키),height,elev,azim].
- 수용: 모든 (subj,cam) 값 산출, NaN 없음.

### D1. [ ] DLT 검증 (SUMediPose: DLT vs 제공calib)
- SUMediPose는 GT 2D-3D(DLT)와 제공 extrinsics 둘 다 → **카메라중심 DLT vs 제공** 거리오차(cm)·각도오차 보고.
- 수용: 오차 작음(예 <수 cm/도) → DLT 신뢰 입증 → AIHub/REHAB 추정 정당화.
- 산출: 검증 표.

### D2. [ ] SUMediPose 통제실험 (기하↔오차)
- 보유 카메라별 raw MAE (C1 12.6/C2 8.6/C3 12.7/C4 12.2/C5 9.4/C6 19.7) ↔ D0의 거리·높이·elev.
- **상관계수 + 산점도**. (같은26명·같은스쿼트, 카메라만 변화 = 통제)
- 수용: 거리/높이↔오차 상관·방향 보고.
- 산출: `sumedipose_geom_vs_error.csv` + 산점도 png.

### D3. [ ] 교란변수 분리
- 오차 ~ 거리 + 높이 + elev + 평균가시성 다중회귀/부분상관 → 주 기여변수.
- 점 수 부족하면(6캠) AIHub 8캠 등 풀링(카메라 단위 다수점).
- 산출: 변수 기여 표.

### D4. [ ] cross-dataset 그룹 검증
- 데이터셋별 평균 카메라 거리(정규화)·높이·elev 집계 → "AIHub/FiT3D 장·고 vs SUMediPose/REHAB 근·저" 가설 **수치 확인**.
- 전이매트릭스(보유: AIHub→REHAB −20%, →FiT3D +10.6%, SUMedi→REHAB +16.3% 등) + z-시프트(a_z AIHub+0.28 vs REHAB+0.08)와 **일관성**.
- 산출: 데이터셋 기하 표 + 종합 그림.

### D5. [ ] 종합
- "카메라기하 유사→전이○, 다름→×" 그림/문장. limitation(REHAB 단위·노이즈) 명시.

---

## 선행 확인 (D0 전)
1. REHAB/SUMediPose 3D 단위(mm/m) + 좌표계 축(어느 게 수직).
2. 관절 대응: AIHub local_kp vs 3d_points 인덱스, REHAB 2d(26) vs 3d(26), SUMediPose xy vs xyz(63) 동일순서?
3. 프레임 정합: 각 셋 2D-3D 프레임 일치(특히 AIHub cam별 vs 세션3d, REHAB 20765 일치 확인됨).
4. 피험자 키: SUMediPose subject_info(height), 나머지 GT3D에서 추정(머리-발 거리).

## 실행 순서
선행확인 → D0(추출기) → D1(검증) → D2(통제·핵심) → D3 → D4 → D5
- **최소 핵심**: D0+D1+D2 (SUMediPose 통제증명 = 가장 강함).
- 전체: +D3,D4,D5 (cross-dataset, 근사 한계명시).

## 한계 (정직)
- AIHub/REHAB intrinsics 없음 → DLT로 추정(검증으로 신뢰도 보강, but FiT3D/SUMediPose만 제공calib 대조 가능).
- cross-dataset 거리 절대값은 단위·정규화 민감 → **높이·elev·상대순서** 중심 해석.
- REHAB GT 2D는 있으나 z-시프트가 이미 핵심증거 → 보조.

관련: `ANALYSIS_PLAN_KR.md`(D 항목), `PAPER_OUTLINE_KR.md`
