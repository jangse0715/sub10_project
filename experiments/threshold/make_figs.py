import pandas as pd, glob, re, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
F='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(F); plt.rcParams['font.family']=fm.FontProperties(fname=F).get_name()
tables, out = sys.argv[1], sys.argv[2]
rows=[]
for p in glob.glob(f'{tables}/*_keypoints.csv'):
    st=p.split('/')[-1][:-14]; m=re.match(r'(S\d\d)_(A\d{3})_(near|far)_(front|left|right)$',st)
    if not m or m.group(4)=='front': continue
    k=pd.read_csv(p); k=k[k.keypoint_name.str.contains('elbow|wrist')].copy(); k['visible']=k.keypoint_name.str.startswith(m.group(4)); rows.append(k)
k=pd.concat(rows); h=k[~k.visible]; v=k[k.visible]
BLUE,ORANGE,INK,MUTED='#2a78d6','#eb6834','#2b2b2a','#8a8983'
def style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#c9c8c2')
    ax.tick_params(colors=MUTED); ax.grid(axis='y',color='#ecebe6',lw=0.8); ax.set_axisbelow(True)
bins=np.arange(0,1.0001,0.05)
fig,ax=plt.subplots(figsize=(8,4.2),dpi=150)
for d,c,lab in [(v,BLUE,'카메라 쪽 팔꿈치·손목'),(h,ORANGE,'반대쪽(가려진) 팔꿈치·손목')]:
    cnt,_=np.histogram(d.confidence.clip(0,0.9999),bins); ax.step(bins[:-1],cnt/cnt.sum()*100,where='post',color=c,lw=2,label=lab)
top=ax.get_ylim()[1]
for t,yy in [(0.5,0.6),(0.8,0.6)]:
    ax.axvline(t,color=MUTED,ls='--',lw=1); ax.text(t+0.01,top*yy,f'{t:.2f}',color=MUTED,fontsize=9)
ax.set_xlabel('관절 신뢰도',color=INK); ax.set_ylabel('관절 비율 (%)',color=INK); ax.set_xlim(0,1)
ax.set_title('옆모습 영상의 팔 관절 신뢰도 분포 (S01·S02, 3동작)',color=INK,fontsize=11,loc='left')
style(ax); ax.legend(frameon=False,loc='upper left',fontsize=9)
fig.tight_layout(); fig.savefig(f'{out}/fig1_arm_conf_distribution.png'); plt.close(fig)
ts=np.round(np.arange(0.25,0.951,0.05),2)
fig,ax=plt.subplots(figsize=(8,4.2),dpi=150)
dk=[(h.confidence<t).mean()*100 for t in ts]; kv=[(v.confidence>=t).mean()*100 for t in ts]
ax.plot(ts,kv,color=BLUE,lw=2,marker='o',ms=5); ax.plot(ts,dk,color=ORANGE,lw=2,marker='s',ms=5)
ax.text(ts[-1]+0.015,kv[-1],'카메라 쪽 팔: 남는 비율',color=INK,fontsize=9,va='center')
ax.text(ts[-1]+0.015,dk[-1],'가려진 팔: 걸러지는 비율',color=INK,fontsize=9,va='center')
for t in [0.5,0.8]: ax.axvline(t,color=MUTED,ls='--',lw=1)
ax.set_xlim(0.22,1.25); ax.set_ylim(0,105); ax.set_xticks(ts[::2])
ax.set_xlabel('기준값 (threshold)',color=INK); ax.set_ylabel('%',color=INK)
ax.set_title('기준값을 올릴 때 남는 관절과 걸러지는 관절 (옆모습, 팔)',color=INK,fontsize=11,loc='left'); style(ax)
fig.tight_layout(); fig.savefig(f'{out}/fig2_threshold_sweep_arms.png'); plt.close(fig)
