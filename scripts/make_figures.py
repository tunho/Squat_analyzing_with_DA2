"""논문 그림 5종 생성 → outputs/paper/figures/"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("outputs/paper/figures",exist_ok=True)
def mae(a,b): return float(np.mean(np.abs(a-b)))
def src(s):
    s=str(s);return 'SUMediPose' if s.startswith('SUME_') else ('AIHub' if s[:2] in('CA','CB','CI') else ('REHAB' if s.startswith('PM') else 'FiT3D'))
DS=['AIHub','SUMediPose','REHAB','FiT3D']
p=pd.read_csv('outputs/paper/pooled4_offstd_loso/loso_predictions.csv'); p['ds']=p.subject_id.map(src)

# Fig1: 데이터셋별 raw vs corrected (전체+deep)
fig,ax=plt.subplots(figsize=(7,4))
x=np.arange(len(DS)); w=0.35
raw=[mae(p[p.ds==d].gt_angle,p[p.ds==d].mp_knee_angle) for d in DS]
cor=[mae(p[p.ds==d].gt_angle,p[p.ds==d].corrected_angle) for d in DS]
ax.bar(x-w/2,raw,w,label='raw MediaPipe',color='#c44')
ax.bar(x+w/2,cor,w,label='corrected (ours)',color='#48a')
for i,(r,c) in enumerate(zip(raw,cor)): ax.text(i,max(r,c)+0.3,f"+{(r-c)/r*100:.0f}%",ha='center',fontsize=9)
ax.set_xticks(x);ax.set_xticklabels(DS);ax.set_ylabel('knee-angle MAE (deg)');ax.legend();ax.set_title('Per-dataset correction (pooled LOSO)')
plt.tight_layout();plt.savefig("outputs/paper/figures/fig1_per_dataset.png",dpi=150);plt.close()

# Fig2: 베이스라인 비교(전체)
b=pd.read_csv('outputs/paper/baselines/pooled4_offstd_baselines.csv'); br=b[b.scope=='전체'].iloc[0]
methods=['raw','affine','calib','linreg','ours']; mae_v=[br[m] for m in methods]
fig,ax=plt.subplots(figsize=(6,4))
cols=['#999','#bbb','#caa','#88c','#48a']
ax.bar(methods,mae_v,color=cols)
for i,v in enumerate(mae_v): ax.text(i,v+0.2,f"{v:.1f}",ha='center',fontsize=9)
ax.set_ylabel('MAE (deg)');ax.set_title('Baseline comparison (pooled LOSO, 84 subj)')
plt.tight_layout();plt.savefig("outputs/paper/figures/fig2_baselines.png",dpi=150);plt.close()

# Fig3: 전이 매트릭스 히트맵
M=np.array([[np.nan,10.7,-18.6,5.3],[0.7,np.nan,15.9,28.9]])
fig,ax=plt.subplots(figsize=(6,2.8))
im=ax.imshow(M,cmap='RdYlGn',vmin=-25,vmax=30,aspect='auto')
ax.set_xticks(range(4));ax.set_xticklabels(DS);ax.set_yticks([0,1]);ax.set_yticklabels(['AIHub→','SUMediPose→'])
for i in range(2):
    for j in range(4):
        if not np.isnan(M[i,j]): ax.text(j,i,f"{M[i,j]:+.1f}%",ha='center',va='center',fontsize=9)
ax.set_title('Naive single-source transfer (improvement %)');plt.colorbar(im,fraction=0.04)
plt.tight_layout();plt.savefig("outputs/paper/figures/fig3_transfer.png",dpi=150);plt.close()

# Fig4: 카메라 균등화 (raw vs corr std)
eq={'AIHub':(4.46,1.98),'SUMediPose':(3.73,1.14),'REHAB':(1.30,0.75),'FiT3D':(6.26,3.41)}
fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(len(DS));w=0.35
ax.bar(x-w/2,[eq[d][0] for d in DS],w,label='raw',color='#c44')
ax.bar(x+w/2,[eq[d][1] for d in DS],w,label='corrected',color='#48a')
ax.set_xticks(x);ax.set_xticklabels(DS);ax.set_ylabel('cross-camera MAE std (deg)')
ax.legend();ax.set_title('Correction equalizes per-camera error (in-domain)')
plt.tight_layout();plt.savefig("outputs/paper/figures/fig4_camera_equalize.png",dpi=150);plt.close()

# Fig5: 각도 trace (SUMediPose 연속 클립 — strided 아님)
import pickle
from paper_train_loso import prepare_v6_data
d5=pd.read_csv('experiments/paper/features_sumedipose_offline.csv')
g5=d5[(d5.subject_id=='S1')&(d5.camera=='C1')].reset_index(drop=True)
fi=g5.frame_index.values; brk=np.where(np.diff(fi)<=0)[0]
end=brk[0]+1 if len(brk) else len(g5)
clip=g5.iloc[:end].copy()
if len(clip)>400: clip=clip.iloc[:400]
feat=prepare_v6_data(clip,'enhanced_safe')
m5=pickle.load(open('outputs/paper/sumedipose_offstd_loso/v6_single_residual_model.pkl','rb'))
corr5=feat.mp_knee_angle+m5['model'].predict(m5['scaler'].transform(feat[m5['features']]))
fig,ax=plt.subplots(figsize=(8,3.5))
ax.plot(feat.gt_angle.values,label='GT (Vicon)',color='k',lw=2)
ax.plot(feat.mp_knee_angle.values,label='raw MediaPipe',color='#c44',alpha=.75)
ax.plot(corr5.values,label='corrected',color='#48a',alpha=.85)
ax.set_xlabel('frame');ax.set_ylabel('knee angle (deg)');ax.legend();ax.set_title('Knee-angle trace (SUMediPose S1/C1, one rep)')
plt.tight_layout();plt.savefig("outputs/paper/figures/fig5_trace.png",dpi=150);plt.close()
print("저장 완료:", os.listdir("outputs/paper/figures"))
