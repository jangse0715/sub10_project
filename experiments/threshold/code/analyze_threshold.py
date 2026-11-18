"""관절 신뢰도 기준값(threshold) 분석.

입력: reports/day02/tables/ 의 10번 결과 (S0x_A00x_<거리>_<방향>_keypoints.csv, _frame_quality.csv)
출력: experiments/threshold/tables/ 아래 표(csv). 그림은 make_figs.py가 만든다.

실행 예 (sub10_project 루트에서)
  python experiments/threshold/code/analyze_threshold.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(r"^(S\d{2})_(A\d{3})_(near|far)_(front|left|right)$")

BODY12 = [f"{s}_{j}" for j in ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
          for s in ["left", "right"]]
FACE5 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

# 분석용 임시 중요 관절 (팀 결정 전 가정)
CRITICAL = {
    "A003": BODY12,  # walk_soothe: 안고 달래며 걷기 → 팔 + 다리
    "A007": BODY12,  # hold_pick_object: 안은 채 바닥까지 앉기 → 팔 + 다리
    "A008": [f"{s}_{j}" for j in ["shoulder", "elbow", "wrist", "hip"]
             for s in ["left", "right"]],  # shake: 팔 + 골반
}
THRESHOLDS = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
BINS = [0.0, 0.25, 0.50, 0.70, 0.90, 1.0001]
BIN_LABELS = ["0~0.25", "0.25~0.50", "0.50~0.70", "0.70~0.90", "0.90~1.00"]


def load(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    kps, fqs = [], []
    for path in sorted(tables.glob("*_keypoints.csv")):
        stem = path.name[: -len("_keypoints.csv")]
        m = NAME.match(stem)
        if not m:
            continue  # 방향 구분 없는 연속 영상 등은 제외
        subject, action, distance, direction = m.groups()
        kp = pd.read_csv(path, encoding="utf-8-sig")
        fq = pd.read_csv(tables / f"{stem}_frame_quality.csv", encoding="utf-8-sig")
        for df in (kp, fq):
            df["video"], df["subject"], df["action"] = stem, subject, action
            df["distance"], df["direction"] = distance, direction
        kps.append(kp)
        fqs.append(fq)
    return pd.concat(kps, ignore_index=True), pd.concat(fqs, ignore_index=True)


def side_label(row) -> str:
    """옆모습 영상에서 카메라 쪽(near)·반대쪽(far) 몸 구분. 정면은 front."""
    if row.direction == "front" or row.keypoint_name in FACE5:
        return "front_or_face"
    side = "left" if row.keypoint_name.startswith("left") else "right"
    return "camera_side" if side == row.visible_side else "hidden_side"


def jitter(kp: pd.DataFrame) -> pd.DataFrame:
    """좌표 흔들림 지표: 앞뒤 2 Frame과의 시간 중앙값에서 떨어진 거리 ÷ 몸통 길이."""
    out = []
    for video, g in kp.groupby("video"):
        x = g.pivot(index="frame_id", columns="keypoint_name", values="pixel_x").sort_index()
        y = g.pivot(index="frame_id", columns="keypoint_name", values="pixel_y").sort_index()
        c = g.pivot(index="frame_id", columns="keypoint_name", values="confidence").sort_index()
        sx = (x["left_shoulder"] + x["right_shoulder"]) / 2
        sy = (y["left_shoulder"] + y["right_shoulder"]) / 2
        hx = (x["left_hip"] + x["right_hip"]) / 2
        hy = (y["left_hip"] + y["right_hip"]) / 2
        torso = float(np.nanmedian(np.hypot(sx - hx, sy - hy)))
        mx = x.rolling(5, center=True).median()
        my = y.rolling(5, center=True).median()
        r = np.hypot(x - mx, y - my) / torso
        long = pd.DataFrame({
            "residual": r.stack(),
            "confidence": c.stack(),
        }).reset_index()
        long["video"] = video
        out.append(long)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="reports/day02/tables")
    ap.add_argument("--out", default="experiments/threshold/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kp, fq = load(Path(args.tables))
    meta = kp[["video", "subject", "action", "distance", "direction"]].drop_duplicates()

    # 옆모습 영상에서 카메라 쪽 판정: 왼쪽·오른쪽 몸 관절 평균 신뢰도가 높은 쪽
    body = kp[kp.keypoint_name.isin(BODY12)].copy()
    body["side"] = np.where(body.keypoint_name.str.startswith("left"), "left", "right")
    vis = body.groupby(["video", "side"]).confidence.mean().unstack()
    vis["visible_side"] = np.where(vis["left"] >= vis["right"], "left", "right")
    kp = kp.merge(vis[["visible_side"]], left_on="video", right_index=True)
    kp["side_group"] = kp.apply(side_label, axis=1)

    # 1) 영상별 요약 (2일차 비교표 형식)
    rows = []
    for v, g in kp.groupby("video"):
        f = fq[fq.video == v]
        m = meta[meta.video == v].iloc[0]
        b = g[g.keypoint_name.isin(BODY12)]
        lowest = g.loc[g.groupby("frame_id").confidence.idxmin(), "keypoint_name"]
        rows.append({
            "video": v, "subject": m.subject, "action": m.action,
            "distance": m.distance, "direction": m.direction,
            "frames": len(f),
            "no_person": int((f.people == 0).sum()),
            "multi_person": int((f.people > 1).sum()),
            "mean_conf_all17": round(g.confidence.mean(), 3),
            "mean_conf_body12": round(b.confidence.mean(), 3),
            "low_ratio_all17_lt050": round((g.confidence < 0.5).mean(), 3),
            "low_ratio_body12_lt050": round((b.confidence < 0.5).mean(), 3),
            "most_often_lowest": lowest.value_counts().index[0],
            "visible_side": g.visible_side.iloc[0] if m.direction != "front" else "-",
            "camera_side_conf": round(g[g.side_group == "camera_side"].confidence.mean(), 3)
            if m.direction != "front" else np.nan,
            "hidden_side_conf": round(g[g.side_group == "hidden_side"].confidence.mean(), 3)
            if m.direction != "front" else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values(["direction", "distance", "action", "subject"])
    keys = ["direction", "distance", "subject"]
    cond = summary.groupby(keys).agg(
        videos=("video", "size"), frames=("frames", "sum"), multi_person=("multi_person", "sum"),
        mean_conf_all17=("mean_conf_all17", "mean"), mean_conf_body12=("mean_conf_body12", "mean"),
        low_all17_lt050=("low_ratio_all17_lt050", "mean"), low_body12_lt050=("low_ratio_body12_lt050", "mean"),
        camera_side_conf=("camera_side_conf", "mean"), hidden_side_conf=("hidden_side_conf", "mean")).round(3)
    cond["most_often_lowest"] = summary.groupby(keys).most_often_lowest.agg(
        lambda x: x.value_counts().index[0])
    cond.to_csv(out / "summary_by_condition.csv", encoding="utf-8-sig")

    # 2) 기준값 구간별 비교 (영상마다)
    sweep = []
    for v, g in kp.groupby("video"):
        m = meta[meta.video == v].iloc[0]
        f = fq[fq.video == v]
        crit = CRITICAL[m.action]
        cg = g[g.keypoint_name.isin(crit)]
        wide = cg.pivot(index="frame_id", columns="keypoint_name", values="confidence")
        frames_total = len(f)
        pose_ratio = (f.people > 0).mean()
        single_ratio = (f.people == 1).mean()
        mean_conf = g.confidence.mean()
        for t in THRESHOLDS:
            low_crit = float((cg.confidence < t).mean())
            check = (pose_ratio < 0.90 or single_ratio < 0.90
                     or mean_conf < 0.50 or low_crit > 0.20)
            sweep.append({
                "video": v, "subject": m.subject, "action": m.action,
                "distance": m.distance, "direction": m.direction, "threshold": t,
                "kept_ratio_body12": round(float((g[g.keypoint_name.isin(BODY12)].confidence >= t).mean()), 3),
                "usable_frame_ratio": round(float((wide >= t).all(axis=1).sum() / frames_total), 3),
                "low_critical_ratio": round(low_crit, 3),
                "qa_status_27": "CHECK" if check else "AUTO_PASS",
                "check_reason_other_than_low": "single_person" if single_ratio < 0.90 else "",
            })
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "threshold_sweep_by_video.csv", index=False, encoding="utf-8-sig")

    grp = (sweep.groupby(["direction", "distance", "threshold"])
           [["kept_ratio_body12", "usable_frame_ratio", "low_critical_ratio"]].mean().round(3))
    grp["videos"] = sweep.groupby(["direction", "distance", "threshold"]).size()
    grp["check_count"] = sweep.groupby(["direction", "distance", "threshold"]).qa_status_27.apply(
        lambda s: int((s == "CHECK").sum()))
    grp.to_csv(out / "threshold_sweep_by_condition.csv", encoding="utf-8-sig")

    # 3) 신뢰도 구간별 좌표 흔들림
    jit = jitter(kp[kp.keypoint_name.isin(BODY12)])
    jit = jit.merge(meta, on="video").merge(
        kp[["video", "frame_id", "keypoint_name", "side_group"]],
        on=["video", "frame_id", "keypoint_name"])
    jit = jit.dropna(subset=["residual"])
    jit["bin"] = pd.cut(jit.confidence, BINS, labels=BIN_LABELS, right=False)
    jb = jit.groupby("bin", observed=False).residual.agg(
        count="size", median="median",
        p90=lambda s: s.quantile(0.9),
        jump_ratio=lambda s: (s > 0.10).mean()).round(3)
    jb.to_csv(out / "jitter_by_conf_bin.csv", encoding="utf-8-sig")

    # 4) 옆모습에서 기준값이 '보이는 관절 / 가려진 관절'을 얼마나 가르나
    side = kp[(kp.direction != "front") & kp.keypoint_name.isin(BODY12)].copy()
    side["visible"] = side.side_group == "camera_side"
    arm = side[side.keypoint_name.str.contains("elbow|wrist")]
    sep = []
    for t in THRESHOLDS:
        sep.append({
            "threshold": t,
            "body12_camera_side_kept": round(float((side[side.visible].confidence >= t).mean()), 3),
            "body12_hidden_side_dropped": round(float((side[~side.visible].confidence < t).mean()), 3),
            "arm_camera_side_kept": round(float((arm[arm.visible].confidence >= t).mean()), 3),
            "arm_hidden_side_dropped": round(float((arm[~arm.visible].confidence < t).mean()), 3),
        })
    pd.DataFrame(sep).to_csv(out / "visibility_separation.csv", index=False, encoding="utf-8-sig")
    hidden = side[~side.visible].copy()
    hidden["joint"] = hidden.keypoint_name.str.replace("left_|right_", "", regex=True)
    (hidden.groupby("joint").confidence.median().round(3)
     .to_csv(out / "hidden_side_median_by_joint.csv", encoding="utf-8-sig"))
    ha = arm[~arm.visible]
    pd.DataFrame({
        "group": ["all", "near", "far", "S01_near", "S02_near", "A003", "A007", "A008"],
        "hidden_arm_median": [
            ha.confidence.median(),
            ha[ha.distance == "near"].confidence.median(),
            ha[ha.distance == "far"].confidence.median(),
            ha[(ha.subject == "S01") & (ha.distance == "near")].confidence.median(),
            ha[(ha.subject == "S02") & (ha.distance == "near")].confidence.median(),
            ha[ha.action == "A003"].confidence.median(),
            ha[ha.action == "A007"].confidence.median(),
            ha[ha.action == "A008"].confidence.median(),
        ],
    }).round(3).to_csv(out / "hidden_arm_median_by_group.csv", index=False, encoding="utf-8-sig")

    # 5) 교재 2일차 §17 비교표 형식 (영상별·조건별) + §24 가장 불안정한 Frame
    tb, worst = [], []
    for v, f in fq.groupby("video"):
        m = meta[meta.video == v].iloc[0]
        g = kp[kp.video == v]
        ext = g.groupby("frame_id").agg(x0=("pixel_x", "min"), x1=("pixel_x", "max"),
                                        y0=("pixel_y", "min"), y1=("pixel_y", "max"))
        area = float((((ext.x1 - ext.x0) * (ext.y1 - ext.y0)) / (720 * 1280)).median())
        tb.append({
            "video": v, "subject": m.subject, "action": m.action,
            "distance": m.distance, "direction": m.direction, "frames": len(f),
            "single_person_ratio": round(float((f.people == 1).mean()), 3),
            "multi_person_frames": int((f.people > 1).sum()),
            "kp_extent_area_ratio": round(area, 3),
            "mean_kp_conf": round(float(f.mean_keypoint_conf.mean()), 3),
            "low_count_mean": round(float(f.low_keypoint_count.mean()), 2),
            "low_count_max": int(f.low_keypoint_count.max()),
            "lowest_keypoint": f.lowest_keypoint.value_counts().index[0],
            "lowest_conf_min": round(float(f.lowest_keypoint_conf.min()), 4),
        })
        w = f.sort_values(["low_keypoint_count", "mean_keypoint_conf"], ascending=[False, True]).iloc[0]
        low_names = g[(g.frame_id == w.frame_id) & (g.confidence < 0.5)].keypoint_name.tolist()
        worst.append({"video": v, "frame_id": int(w.frame_id), "time_sec": w.time_sec,
                      "low_count": int(w.low_keypoint_count),
                      "mean_kp_conf": round(float(w.mean_keypoint_conf), 4),
                      "lowest_keypoint": w.lowest_keypoint, "lowest_conf": w.lowest_keypoint_conf,
                      "low_keypoints": " ".join(low_names)})
    tb = pd.DataFrame(tb)
    tb.to_csv(out / "textbook_format_by_video.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(worst).to_csv(out / "worst_frame_by_video.csv", index=False, encoding="utf-8-sig")
    tc = tb.groupby(keys).agg(
        videos=("video", "size"), frames=("frames", "sum"),
        single_person_ratio=("single_person_ratio", "mean"), multi_person_frames=("multi_person_frames", "sum"),
        kp_extent_area_ratio=("kp_extent_area_ratio", "mean"), mean_kp_conf=("mean_kp_conf", "mean"),
        low_count_mean=("low_count_mean", "mean"), low_count_max=("low_count_max", "max"),
        lowest_conf_min=("lowest_conf_min", "min")).round(3)
    tc["lowest_keypoint"] = tb.groupby(keys).lowest_keypoint.agg(lambda x: x.value_counts().index[0])
    tc.to_csv(out / "textbook_format_by_condition.csv", encoding="utf-8-sig")

    # 6) 정면 영상에서 신뢰도가 높은데 좌우가 뒤집힌 Frame
    flips = []
    for v, g in kp[kp.direction == "front"].groupby("video"):
        X = g.pivot(index="frame_id", columns="keypoint_name", values="pixel_x")
        C = g.pivot(index="frame_id", columns="keypoint_name", values="confidence")
        facing = (X.left_shoulder > X.right_shoulder)  # 카메라를 보면 사람의 왼쪽이 화면 오른쪽
        major = facing.mean() >= 0.5
        for j in ["shoulder", "hip", "knee", "ankle"]:
            order = X[f"left_{j}"] > X[f"right_{j}"]
            hi = (C[f"left_{j}"] >= 0.8) & (C[f"right_{j}"] >= 0.8)
            bad = (order != major) & hi
            for fid in bad[bad].index:
                flips.append({"video": v, "frame_id": int(fid), "joint": j,
                              "left_conf": round(float(C.loc[fid, f"left_{j}"]), 3),
                              "right_conf": round(float(C.loc[fid, f"right_{j}"]), 3)})
    pd.DataFrame(flips, columns=["video", "frame_id", "joint", "left_conf", "right_conf"]).to_csv(
        out / "front_left_right_flips.csv", index=False, encoding="utf-8-sig")

    # 7) 골격 모양 검사: 정면 어깨 폭 붕괴, 기록된 골격이 배우 위치에서 튀는지
    shape = []
    for v, g in kp.groupby("video"):
        m = meta[meta.video == v].iloc[0]
        X = g.pivot(index="frame_id", columns="keypoint_name", values="pixel_x")
        Y = g.pivot(index="frame_id", columns="keypoint_name", values="pixel_y")
        C = g.pivot(index="frame_id", columns="keypoint_name", values="confidence")
        b = g[g.keypoint_name.isin(BODY12)]
        cx = b.groupby("frame_id").pixel_x.mean()
        cy = b.groupby("frame_id").pixel_y.mean()
        jump = np.hypot(cx - cx.rolling(9, center=True, min_periods=1).median(),
                        cy - cy.rolling(9, center=True, min_periods=1).median())
        row = {"video": v, "direction": m.direction, "track_jump_frames_gt100px": int((jump > 100).sum())}
        if m.direction == "front":
            sw = np.hypot(X.left_shoulder - X.right_shoulder, Y.left_shoulder - Y.right_shoulder)
            hw = np.hypot(X.left_hip - X.right_hip, Y.left_hip - Y.right_hip)
            hi = (C.left_shoulder >= 0.8) & (C.right_shoulder >= 0.8)
            row.update({
                "shoulder_w_p90_px": round(float(sw.quantile(0.9))),
                "shoulder_collapsed_frames": int(((sw < 0.6 * sw.quantile(0.9)) & hi).sum()),
                "hip_w_median_px": round(float(hw.median())),
                "hip_w_min_px": round(float(hw.min())),
            })
        shape.append(row)
    pd.DataFrame(shape).to_csv(out / "skeleton_shape_checks.csv", index=False, encoding="utf-8-sig")

    print("done:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
