"""사람이 2명 이상 잡힌 영상에서, 두 번째 이후 검출(배우가 아닌 대상)의 신뢰도를 기록한다.

10번과 같은 모델·같은 사람 검출 기준(project.json)을 쓴다.
sub10_project 루트에서 실행해야 모델 파일(yolo11n-pose.pt)을 프로젝트 것으로 쓴다.

실행 예
  python experiments/threshold/code/check_secondary_detections.py \
      --out experiments/threshold/tables/secondary_detections.csv \
      "<원본 영상 경로 1>" "<원본 영상 경로 2>" ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default="configs/project.json")
    ap.add_argument("--out", default="experiments/threshold/tables/secondary_detections.csv")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = YOLO(cfg["pose_model"])
    rows = []
    for path in args.videos:
        cap = cv2.VideoCapture(path)
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1  # 10번과 같이 1부터 센다
            r = model.predict(frame, imgsz=cfg["image_size"], conf=cfg["person_confidence"], verbose=False)[0]
            if r.boxes is None or len(r.boxes) < 2:
                continue
            box_conf = r.boxes.conf.cpu().numpy()
            kp_conf = r.keypoints.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for rank, i in enumerate(np.argsort(-box_conf)):
                rows.append({
                    "video": Path(path).stem, "frame_id": frame_id, "rank": rank,
                    "box_conf": round(float(box_conf[i]), 3),
                    "kp_mean_all17": round(float(kp_conf[i].mean()), 3),
                    "kp_mean_body12": round(float(kp_conf[i][5:].mean()), 3),
                    "kp_count_ge050": int((kp_conf[i] >= 0.5).sum()),
                    "box_y1": round(float(xyxy[i][1])), "box_h": round(float(xyxy[i][3] - xyxy[i][1])),
                })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    sec = df[df["rank"] == 1]
    print(sec.groupby("video").agg(
        frames=("frame_id", "size"), box_conf=("box_conf", "mean"),
        kp_body12=("kp_mean_body12", "mean"), kp_ge050=("kp_count_ge050", "mean"),
        kp_ge050_max=("kp_count_ge050", "max")).round(2))


if __name__ == "__main__":
    main()
