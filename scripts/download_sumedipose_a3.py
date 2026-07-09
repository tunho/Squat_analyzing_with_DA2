"""SUMediPose (Harvard Dataverse doi:10.7910/DVN/GRRROM) → A3=Squat subset 전체 다운로드.
사용자 선택 로직 그대로 + 병렬 다운로드(재개=기존파일 skip, 재시도, .part 임시).
A3 스쿼트: 6캠(C1~C6) × 26명 × 테이크 + WCS(3D GT) + internal(2D) + calib + metadata ≈ 124.5GB.
"""
import json, re, time, sys, os
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

SERVER_URL = "https://dataverse.harvard.edu"
PERSISTENT_ID = "doi:10.7910/DVN/GRRROM"
VERSION = "1.0"
# repo-relative; override via env EXE_EST_ROOT
OUT_DIR = Path(os.environ.get("EXE_EST_ROOT", Path(__file__).resolve().parent.parent)) / "dataset/raw/sumedipose_a3"
SQUAT_ACTION = "A3"
CAMERAS = [f"C{i}" for i in range(1, 7)]
SUBJECTS = ["S1","S2","S3","S4","S6","S7","S8","S9","S11","S12","S13","S14","S15","S16",
            "S17","S18","S19","S20","S21","S22","S23","S24","S25","S26","S27","S28"]
METADATA_FILES = {"markers_points.json","subject_info.tab","SUMediPose.csv",
                  "vicon_notes.xlsx","external_calibration.tab","subject_map.json"}
CALIB = re.compile(r"^SS.*_external_calibration.*\.npz$")
WORKERS = 6

def required_cat(fn, d):
    if fn in METADATA_FILES: return "metadata"
    if d == "calibration/external/matrix" and CALIB.match(fn): return "calib"
    if d == "WCS" and fn in {f"{s}.zip" for s in SUBJECTS}: return "WCS"
    if d.startswith("internal_data/"):
        p=d.split("/")
        if len(p)==2 and p[1] in CAMERAS and fn in {f"{s}.zip" for s in SUBJECTS}: return "internal"
    if d.startswith("frames/") and fn.endswith(".zip"):
        p=d.split("/")
        if len(p)==3 and p[1] in CAMERAS and p[2] in SUBJECTS \
           and re.match(rf"^{p[1]}{p[2]}{SQUAT_ACTION}D[123]\.zip$", fn): return "frames"
    return None

def fetch_manifest():
    url=f"{SERVER_URL}/api/datasets/:persistentId/versions/{VERSION}?persistentId={quote(PERSISTENT_ID, safe='')}"
    r=requests.get(url, timeout=120); r.raise_for_status()
    sel=[]
    for it in r.json()["data"]["files"]:
        df=it["dataFile"]; d=it.get("directoryLabel","")
        cat=required_cat(df["filename"], d)
        if cat:
            path=f"{d}/{df['filename']}" if d else df["filename"]
            sel.append({"id":df["id"],"path":path,"size":df.get("filesize",0),"cat":cat})
    return sel

_lock=threading.Lock()
_done=0; _bytes=0

def download(f, total_n, total_bytes, retries=5):
    global _done, _bytes
    out=OUT_DIR/f["path"]; out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size>0:
        with _lock:
            _done+=1; _bytes+=f["size"]
        return True
    url=f"{SERVER_URL}/api/access/datafile/{f['id']}"
    tmp=out.with_suffix(out.suffix+".part")
    for att in range(1,retries+1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp,"wb") as fh:
                    for ch in r.iter_content(1024*1024):
                        if ch: fh.write(ch)
            tmp.rename(out)
            with _lock:
                _done+=1; _bytes+=f["size"]
                print(f"[{_done}/{total_n}] {_bytes/1e9:.1f}/{total_bytes/1e9:.1f}GB  {f['path']}", flush=True)
            return True
        except Exception as e:
            print(f"[WARN] {f['path']} att{att}/{retries}: {e}", flush=True)
            time.sleep(5*att)
    print(f"[FAIL] {f['path']}", flush=True)
    return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] manifest 조회...", flush=True)
    sel=fetch_manifest()
    total_n=len(sel); total_bytes=sum(f["size"] for f in sel)
    json.dump(sel, open(OUT_DIR/"manifest.json","w"), indent=2, ensure_ascii=False)
    cats={}
    for f in sel: cats[f["cat"]]=cats.get(f["cat"],0)+1
    print(f"[INFO] 선택 {total_n}파일 {total_bytes/1e9:.1f}GB  {cats}", flush=True)
    fails=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs={ex.submit(download,f,total_n,total_bytes):f for f in sel}
        for fu in as_completed(futs):
            if not fu.result(): fails.append(futs[fu]["path"])
    print(f"\n[DONE] {total_n-len(fails)}/{total_n} 성공, 실패 {len(fails)}", flush=True)
    if fails: print("실패목록:", fails[:20], flush=True)

if __name__=="__main__":
    main()
