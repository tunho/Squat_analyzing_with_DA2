"""4셋 카메라 기하 복원 → 카메라별 거리·높이(피험자키 정규화). DLT(AIHub/REHAB/SUMedi) + 제공calib(FiT3D)."""
import numpy as np, json, zipfile, glob, re, os, pandas as pd
np.seterr(all='ignore')

def dlt_center(p2,p3):
    A=[]
    for (u,v),(X,Y,Z) in zip(p2,p3):
        A.append([X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z,-u])
        A.append([0,0,0,0,X,Y,Z,1,-v*X,-v*Y,-v*Z,-v])
    A=np.array(A,float)
    if len(A)<12 or np.isnan(A).any(): return None
    _,_,Vt=np.linalg.svd(A); P=Vt[-1].reshape(3,4)
    _,_,V2=np.linalg.svd(P); C=V2[-1]
    return C[:3]/C[3]

def geom(C, pts3d, vaxis):
    """카메라중심 C, 피험자 GT3D(전 프레임·관절 Nx3) → 거리/높이(키정규화)·elevation."""
    cen=np.nanmean(pts3d,0)
    height=np.nanmax(pts3d[:,vaxis])-np.nanmin(pts3d[:,vaxis])  # 피험자 키(수직범위)
    d=np.linalg.norm(C-cen)
    h=C[vaxis]-cen[vaxis]
    horiz=np.sqrt(max(d**2-h**2,1e-9))
    elev=np.degrees(np.arctan2(h,horiz))  # 올려/내려본 각
    return d/height, h/height, elev, d, height

rows=[]

# ---- SUMediPose: GT 2D(xy)+3D(xyz), vaxis=1(Y) ----
for sz in sorted(glob.glob("dataset/raw/sumedipose_a3/internal_data/C*/S*.zip")):
    cam='C'+re.search(r'/C(\d)/',sz).group(1); subj=re.search(r'/(S\d+)\.zip',sz).group(1)
    z=zipfile.ZipFile(sz); a3=[n for n in z.namelist() if re.match(rf'{subj}A3D\d+\.json',n)]
    if not a3: continue
    d=json.loads(z.read(a3[0]))
    P2=np.array([f['xy'] for f in d]).reshape(-1,2); P3=np.array([f['xyz'] for f in d]).reshape(-1,3)
    m=~(np.isnan(P2).any(1)|np.isnan(P3).any(1)); P2,P3=P2[m],P3[m]
    if len(P2)<100: continue
    idx=np.linspace(0,len(P2)-1,min(800,len(P2))).astype(int)
    C=dlt_center(P2[idx],P3[idx])
    if C is None: continue
    g=geom(C,P3,1)
    rows.append(("SUMediPose",cam,*g))

# ---- REHAB: 2d_joints/3d_joints npy, vaxis=1(Y), m ----
for f2 in sorted(glob.glob("dataset/raw/REHAB24-6/2d_joints/Ex6/*-c1*-30fps.npy")):
    bn=os.path.basename(f2); subj=bn.split('-')[0]; cam=re.search(r'-(c1\d)-',bn).group(1)
    f3=f"dataset/raw/REHAB24-6/3d_joints/Ex6/{subj}-30fps.npy"
    if not os.path.exists(f3): continue
    a2=np.load(f2); a3=np.load(f3)[...,:3]
    n=min(len(a2),len(a3)); a2,a3=a2[:n],a3[:n]
    P2=a2.reshape(-1,2); P3=a3.reshape(-1,3)
    m=~(np.isnan(P2).any(1)|np.isnan(P3).any(1)|(P2==0).all(1)); P2,P3=P2[m],P3[m]
    if len(P2)<100: continue
    idx=np.linspace(0,len(P2)-1,min(800,len(P2))).astype(int)
    C=dlt_center(P2[idx],P3[idx])
    if C is None: continue
    g=geom(C,P3,1)
    rows.append(("REHAB",cam,*g))

# ---- AIHub: 3d_points.csv + local_keypoints csv, vaxis=2(Z), mm ----
def aihub_cols(df, d3=True):
    js=[c[:-2] for c in df.columns if c.endswith('_x')]
    return js
for s3 in glob.glob("dataset/raw/aihub_squat/labels/스쿼트/*/*/*/*/*/3d_points.csv")[:120]:
    sess=os.path.dirname(s3)
    df3=pd.read_csv(s3); js3=aihub_cols(df3)
    for camdir in sorted(glob.glob(sess+"/camera*")):
        cam='camera'+re.search(r'camera(\d)',camdir).group(1)
        lks=glob.glob(camdir+"/local_keypoints/*.csv")
        if not lks: continue
        df2=pd.read_csv(lks[0]); js2=aihub_cols(df2)
        common=[j for j in js3 if j in js2]
        n=min(len(df2),len(df3))
        P3=df3[[f"{j}_{a}" for j in common for a in 'xyz']].to_numpy()[:n].reshape(-1,len(common),3).reshape(-1,3)
        P2=df2[[f"{j}_{a}" for j in common for a in 'xy']].to_numpy()[:n].reshape(-1,len(common),2).reshape(-1,2)
        m=~(np.isnan(P2).any(1)|np.isnan(P3).any(1)); P2,P3=P2[m],P3[m]
        if len(P2)<100: continue
        idx=np.linspace(0,len(P2)-1,min(600,len(P2))).astype(int)
        C=dlt_center(P2[idx],P3[idx])
        if C is None: continue
        g=geom(C,P3,2)
        rows.append(("AIHub",cam,*g))

# ---- FiT3D: 제공 extrinsics, 3D joints3d_25(json), vaxis=? (m) ----
camname={"50591643":"left_front","58860488":"right_rear","60457274":"right_front","65906101":"left_rear"}
for s in ["s03","s04","s05"]:
    base=f"dataset/raw/fit3d/fit3d_train/train/{s}"
    j3=json.load(open(f"{base}/joints3d_25/squat.json"))
    P3=np.array(j3['joints3d_25'] if isinstance(j3,dict) else j3).reshape(-1,3)
    vaxis=int(np.argmax(P3.std(0)))  # 수직축 자동
    for cid,cn in camname.items():
        cp=json.load(open(f"{base}/camera_parameters/{cid}/squat.json"))
        R=np.array(cp['extrinsics']['R']); T=np.array(cp['extrinsics']['T']).reshape(3)
        C=-R.T@T
        g=geom(C,P3,vaxis)
        rows.append(("FiT3D",cn,*g))

out=pd.DataFrame(rows,columns=["dataset","camera","dist_norm","height_norm","elev","dist_raw","subj_height"])
agg=out.groupby(["dataset","camera"]).mean(numeric_only=True).reset_index()
os.makedirs("outputs/paper/camera_geom",exist_ok=True)
agg.to_csv("outputs/paper/camera_geom/camera_positions.csv",index=False)
print(agg.round(2).to_string(index=False))
