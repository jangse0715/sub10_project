"""육안 판정용 표본 시트(figures/spot_sheet.jpg)와 표본 목록(labels/spot_samples.csv)을 만든다.

실행 예 (sub10_project 루트에서)
  python experiments/threshold/code/spot_sample.py
판정 결과(labels/spot_judgments.csv)는 사람이 시트를 보고 직접 적는다.
"""
import cv2, numpy as np, pandas as pd
from pathlib import Path
# sub10_project 루트 기준 경로. 원본 영상은 data/videos/<사람번호>/<영상이름>.mp4 에 있어야 한다.
ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "reports" / "day02" / "tables"
VIDEO_DIR = ROOT / "data" / "videos"
NAMES = ['S01_A003_near_right','S01_A007_near_right','S01_A003_far_right','S02_A003_near_right','S02_A007_near_right']
V = {n: str(VIDEO_DIR / n[:3] / f"{n}.mp4") for n in NAMES}
BODY=[f"{s}_{j}" for j in ["shoulder","elbow","wrist","hip","knee","ankle"] for s in ["left","right"]]
kp=pd.concat([pd.read_csv(str(TABLES / f'{v}_keypoints.csv')).assign(video=v) for v in V])
kp=kp[kp.keypoint_name.isin(BODY)].copy()
kp['hidden']=kp.keypoint_name.str.startswith('left')
groups=[('H 0.25-0.50',kp[kp.hidden&(kp.confidence>=.25)&(kp.confidence<.5)],8),
        ('H 0.50-0.70',kp[kp.hidden&(kp.confidence>=.5)&(kp.confidence<.7)],8),
        ('H 0.70-0.90',kp[kp.hidden&(kp.confidence>=.7)&(kp.confidence<.9)],4),
        ('C >=0.90',kp[~kp.hidden&(kp.confidence>=.9)],4)]
picked=pd.concat([df.sample(n,random_state=11+k).assign(group=name) for k,(name,df,n) in enumerate(groups)]).reset_index(drop=True)
tiles=[]
for i,r in picked.iterrows():
    cap=cv2.VideoCapture(V[r.video]); cap.set(cv2.CAP_PROP_POS_FRAMES,int(r.frame_id)-1); ok,fr=cap.read()
    x,y=int(r.pixel_x),int(r.pixel_y); R=170
    pad=cv2.copyMakeBorder(fr,R,R,R,R,cv2.BORDER_CONSTANT,value=(40,40,40))
    crop=pad[y:y+2*R,x:x+2*R].copy()
    cv2.circle(crop,(R,R),9,(0,0,255),2); cv2.circle(crop,(R,R),2,(0,0,255),-1)
    opp=r.keypoint_name.replace('left','RIGHT').replace('right','left').replace('RIGHT','right')
    o=kp[(kp.video==r.video)&(kp.frame_id==r.frame_id)&(kp.keypoint_name==opp)].iloc[0]
    ox,oy=int(o.pixel_x)-x+R,int(o.pixel_y)-y+R
    if 0<=ox<2*R and 0<=oy<2*R: cv2.drawMarker(crop,(ox,oy),(0,200,0),cv2.MARKER_CROSS,14,2)
    crop=cv2.resize(crop,(260,260))
    lab=np.full((44,260,3),255,np.uint8)
    cv2.putText(lab,f"#{i} {r.group}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
    cv2.putText(lab,f"{r.keypoint_name} {r.confidence:.2f} {r.video[:3]}{r.video[3:8]} f{int(r.frame_id)}",(4,36),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
    tiles.append(np.vstack([lab,crop]))
cv2.imwrite(str(ROOT / 'experiments' / 'threshold' / 'figures' / 'spot_sheet.jpg'),np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,24,6)]),[cv2.IMWRITE_JPEG_QUALITY,85])
picked[['group','video','frame_id','keypoint_name','confidence','pixel_x','pixel_y']].to_csv(str(ROOT / 'experiments' / 'threshold' / 'labels' / 'spot_samples.csv'),index_label='id')
print(picked[['group','video','frame_id','keypoint_name','confidence']].to_string())
