"""SUMediPose A3=Squat → RAW 덤프 (AIHub 덤프와 동일 스키마, aihub_features_from_dump.py 재사용 가능).
한 번 뽑고 영원히: MediaPipe world 33 + image 33 (x,y,z,v) + Vicon GT 양다리 무릎각 + 메타.
take별 parquet (재개=이미존재 skip).
입력: internal_data/C{cam}/S{subj}.zip 안 S{subj}A3D{take}.json (point_ids,xyz[Vicon3D],path),
      frames/C{cam}/S{subj}/C{cam}S{subj}A3D{take}.zip (jpg).
"""
from __future__ import annotations
import argparse, json, math, re, zipfile, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd, cv2
from squat.pose_estimation import PoseEstimator

NUM_LM=33
LEFT=("LHJC","LKJC","LANK"); RIGHT=("RHJC","RKJC","RANK")

def vang(xyz, ids, joints):
    try: h,k,a=(ids.index(j) for j in joints)
    except ValueError: return float("nan")
    H,K,A=np.array(xyz[h]),np.array(xyz[k]),np.array(xyz[a])
    ba,bc=H-K,A-K
    n=np.linalg.norm(ba)*np.linalg.norm(bc)
    if n<1e-9: return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(ba,bc)/n,-1,1)))

def process_take(pose, frames_zip, take_json, meta, step_ms=33):
    tmp=Path(tempfile.mkdtemp(prefix="sumed_"))
    try:
        with zipfile.ZipFile(frames_zip) as z: z.extractall(tmp)
        recs=[fr for fr in take_json if "path" in fr and "xyz" in fr]
        recs.sort(key=lambda fr: fr.get("frame_num",0))
        ids=recs[0]["point_ids"] if recs else None
        rows=[]
        for i,fr in enumerate(recs):
            jpg=tmp/Path(fr["path"]).name
            if not jpg.exists(): continue
            img=cv2.imread(str(jpg))
            if img is None: continue
            lm,res=pose.extract_keypoints_only(img, frame_timestamp_ms=i*step_ms)
            if not lm or res is None or not getattr(res,"pose_world_landmarks",None): continue
            w=res.pose_world_landmarks[0]
            row=dict(meta); row["frame_index"]=i
            row["gt_knee_left"]=vang(fr["xyz"],ids,LEFT)
            row["gt_knee_right"]=vang(fr["xyz"],ids,RIGHT)
            for j in range(NUM_LM):
                wl=w[j]; vis=getattr(wl,"visibility",None); pr=getattr(wl,"presence",None)
                row[f"w{j}_x"],row[f"w{j}_y"],row[f"w{j}_z"]=float(wl.x),float(wl.y),float(wl.z)
                row[f"w{j}_v"]=float(vis if vis is not None else (pr or 0.0))
                p=lm[j]
                row[f"p{j}_x"],row[f"p{j}_y"],row[f"p{j}_z"],row[f"p{j}_v"]=p["x"],p["y"],p["z"],p["visibility"]
            rows.append(row)
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cameras", nargs="*", default=[f"C{i}" for i in range(1,7)])
    p.add_argument("--subjects", nargs="*", default=None)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done=skip=0
    for cam in args.cameras:
        ezdir=args.root/"internal_data"/cam
        if not ezdir.exists(): continue
        for sz in sorted(ezdir.glob("S*.zip")):
            subject=sz.stem
            if args.subjects and subject not in args.subjects: continue
            with zipfile.ZipFile(sz) as z:
                for jn in sorted(n for n in z.namelist() if re.match(rf"^{subject}A3D\d+\.json$",n)):
                    take=re.search(r"A3D(\d+)",jn).group(1)
                    outp=args.output_dir/f"{cam}_{subject}_A3D{take}.parquet"
                    if outp.exists(): skip+=1; continue
                    fzip=args.root/"frames"/cam/subject/f"{cam}{subject}A3D{take}.zip"
                    if not fzip.exists(): continue
                    meta={"subject_id":subject,"camera":cam,"cam_idx":int(cam[1:]),
                          "view_type":"vicon","ex_type":"squat","take":int(take)}
                    pose=PoseEstimator(static_image_mode=False)
                    try: rows=process_take(pose,fzip,json.loads(z.read(jn)),meta)
                    finally: pose.close()
                    if rows:
                        pd.DataFrame(rows).to_parquet(outp,index=False); done+=1
                        print(f"[{done}] {outp.name}: {len(rows)}행",flush=True)
    print(f"\n완료: {done} take 덤프, {skip} skip",flush=True)

if __name__=="__main__":
    main()
