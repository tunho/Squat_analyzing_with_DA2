"""SUMediPose A3=Squat → official feature CSV (4번째 데이터셋).

구조: internal_data/C{cam}/S{subj}.zip 안 S{subj}A3D{take}.json (point_ids + xyz[3D Vicon GT] + path[jpg])
      frames/C{cam}/S{subj}/C{cam}S{subj}A3D{take}.zip 안 jpg들.
GT 무릎각 = Vicon xyz(LHJC/LKJC/LANK), MediaPipe world(23/25/27)로 mp_knee_angle. path로 정합.
출력 = features_aihub 와 동일 스키마(extract_feature_rows 재사용) → pooled LOSO 편입.

사용:
  PYTHONPATH=../src ../.venv-paper/bin/python sumedipose_extract.py \
    --root ../dataset/raw/sumedipose_a3 --output ../experiments/paper/features_sumedipose_causal.csv \
    --calib-frames 15
"""
from __future__ import annotations
import argparse, json, math, re, zipfile, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd, cv2

from squat.domain.types import Point3D
from squat.pose_estimation import PoseEstimator
from paper_extract import extract_feature_rows, calib_med_total

# MediaPipe world 좌측: hip23 knee25 ankle27 shoulder11
MP = dict(hip=23, knee=25, ankle=27, sho=11)
GT_JOINTS = ("LHJC", "LKJC", "LANK")


class _F:
    __slots__ = ("frame_index","mp_world_hip","mp_world_knee","mp_world_ankle","mp_world_shoulder","shoulder")


def vicon_knee_angle(xyz, ids):
    h,k,a = (ids.index(j) for j in GT_JOINTS)
    H,K,A = np.array(xyz[h]),np.array(xyz[k]),np.array(xyz[a])
    ba,bc = H-K, A-K
    c = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
    return math.degrees(math.acos(np.clip(c,-1,1)))


def process_take(pose, frames_zip: Path, take_json: list, subject, camera, take, calib_frames, step_ms=33):
    tmp = Path(tempfile.mkdtemp(prefix="sume_"))
    try:
        with zipfile.ZipFile(frames_zip) as z:
            z.extractall(tmp)
        recs = [fr for fr in take_json if "path" in fr and "xyz" in fr]
        recs.sort(key=lambda fr: fr.get("frame_num", 0))
        ids = recs[0]["point_ids"] if recs else None
        sframes=[]; gt={}
        for i, fr in enumerate(recs):
            jpg = tmp / Path(fr["path"]).name
            if not jpg.exists(): continue
            img = cv2.imread(str(jpg))
            if img is None: continue
            lm, res = pose.extract_keypoints_only(img, frame_timestamp_ms=i*step_ms)
            if res is None or not getattr(res,"pose_world_landmarks",None): continue
            w = res.pose_world_landmarks[0]
            def P(idx): return Point3D(float(w[idx].x),float(w[idx].y),float(w[idx].z),
                                       float(getattr(w[idx],"visibility",1.0) or 1.0))
            f=_F(); f.frame_index=i
            f.mp_world_hip=P(MP["hip"]); f.mp_world_knee=P(MP["knee"])
            f.mp_world_ankle=P(MP["ankle"]); f.mp_world_shoulder=P(MP["sho"]); f.shoulder=None
            sframes.append(f)
            gt[i]=vicon_knee_angle(fr["xyz"], ids)
        if not sframes: return None
        med = calib_med_total(sframes, calib_frames)
        rows = extract_feature_rows(sframes, f"{camera}", str(subject), med_total=med)
        if not rows: return None
        df = pd.DataFrame(rows)
        df["view_type"]="vicon"; df["camera"]=camera
        df["gt_angle"]=df["frame_index"].map(gt)
        df["ex_type"]="squat"; df["take"]=take
        return df.dropna(subset=["gt_angle"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cameras", nargs="*", default=[f"C{i}" for i in range(1,7)])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--calib-frames", type=int, default=15)
    args=p.parse_args()

    all_df=[]; done=0
    for cam in args.cameras:
        ezdir = args.root/"internal_data"/cam
        if not ezdir.exists(): continue
        subj_zips = sorted(ezdir.glob("S*.zip"))
        for sz in subj_zips:
            subject = sz.stem
            if args.subjects and subject not in args.subjects: continue
            with zipfile.ZipFile(sz) as z:
                jsons = [n for n in z.namelist() if re.match(rf"^{subject}A3D\d+\.json$", n)]
                for jn in sorted(jsons):
                    take = re.search(r"A3D(\d+)", jn).group(1)
                    fzip = args.root/"frames"/cam/subject/f"{cam}{subject}A3D{take}.zip"
                    if not fzip.exists(): continue
                    tj = json.loads(z.read(jn))
                    pose = PoseEstimator(static_image_mode=False)   # take마다 새 인스턴스(타임스탬프 단조 보장)
                    try:
                        df = process_take(pose, fzip, tj, subject, cam, take, args.calib_frames)
                    finally:
                        pose.close()
                    if df is not None and not df.empty:
                        all_df.append(df); done+=1
                        print(f"[{done}] {cam}/{subject}/A3D{take}: {len(df)}행", flush=True)
    if not all_df: raise RuntimeError("추출 결과 없음")
    out=pd.concat(all_df, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n저장: {args.output} ({len(out):,}행, {out['subject_id'].nunique()}명, {done} takes)", flush=True)


if __name__=="__main__":
    main()
