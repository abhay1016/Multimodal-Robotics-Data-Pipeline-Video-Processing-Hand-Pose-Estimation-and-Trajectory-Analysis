"""Task 7: Rerun recording on the common synchronized Ego timeline.

Logs: 4 synced Ego camera streams (JPEG, subsampled), hand keypoints + skeletons, per-frame detection counts
(missing detections visible as 0), VO camera pose/frustum + full path in 3D, tracking inliers, real GoPro IMU
(gyro/accel, ~200 Hz) per camera, and sync offsets/residuals as text.
Usage: python src/visualize_rerun.py [--seconds 60] [--img-fps 10]   ->  outputs/pipeline.rrd  (open: rerun outputs/pipeline.rrd)
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from common import OUT, REPORTS, videos

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
COLORS = {"Left": [255, 80, 80], "Right": [80, 160, 255]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--img-fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()
    sync = json.loads((REPORTS / "sync.json").read_text())
    starts = sync["common_timeline"]["start_in_each_cam_s"]
    vids = videos("ego")
    names = [v.stem[:2] for v in vids]

    rr.init("jupiter_ego_pipeline")
    rr.save(str(OUT / "pipeline.rrd"))
    bp = rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(rrb.Grid(*[rrb.Spatial2DView(origin=f"cams/{n}", name=n) for n in names]),
                       rrb.Spatial3DView(origin="world", name="VO trajectory (HL, arbitrary scale)")),
        rrb.Horizontal(rrb.TimeSeriesView(origin="hands", name="hands detected per frame"),
                       rrb.TimeSeriesView(origin="imu", name="IMU gyro (rad/s)"),
                       rrb.TimeSeriesView(origin="vo", name="VO inliers / tracked")),
        row_shares=[3, 1.3]))
    rr.send_blueprint(bp)
    rr.log("sync_info", rr.TextDocument(json.dumps({k: sync[k] for k in ("method", "cameras", "validation")},
                                                   indent=1), media_type="text/plain"), static=True)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    for v, n in zip(vids, names):
        s0 = starts[v.stem]
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w, h = a.width, int(round(H * a.width / W))
        f0, f1 = int(round(s0 * fps)), int(round((s0 + a.seconds) * fps))
        step = max(1, int(round(fps / a.img_fps)))
        hj = OUT / "hands" / f"{v.stem}_hands.json"
        hands = {fr["frame_index"]: fr for fr in json.loads(hj.read_text())["frames"]} if hj.exists() else {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for fi in range(f0, f1):
            ok, img = cap.read()
            if not ok:
                break
            T = fi / fps - s0
            rr.set_time("common_time", duration=T)
            fr = hands.get(fi)
            if fr is not None:
                rr.log(f"hands/{n}/num_hands", rr.Scalars(fr["num_hands"]))
            if (fi - f0) % step:
                continue
            small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            rr.log(f"cams/{n}/image", rr.EncodedImage(contents=cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes(),
                                                     media_type="image/jpeg"))
            if fr is None:
                continue
            pts, strips, cols = [], [], []
            for hnd in fr["hands"]:
                p = np.array(hnd["landmarks_norm"])[:, :2] * [w, h]
                c = COLORS[hnd["handedness"]]
                pts.append(p)
                cols += [c] * 21
                strips += [[p[i], p[j]] for i, j in HAND_EDGES]
            if pts:
                rr.log(f"cams/{n}/hands/points", rr.Points2D(np.vstack(pts), colors=cols, radii=2.5))
                rr.log(f"cams/{n}/hands/skeleton", rr.LineStrips2D(strips, colors=[200, 255, 0], radii=1.0))
            else:
                rr.log(f"cams/{n}/hands/points", rr.Clear(recursive=False))
                rr.log(f"cams/{n}/hands/skeleton", rr.Clear(recursive=False))
            rr.log(f"cams/{n}/status", rr.TextLog(fr["status"], level="INFO" if fr["hands"] else "WARN"))
        cap.release()

        # real IMU (gyro + accel) as columns on the common timeline
        tag = f"{v.parent.name}_{v.stem}"
        for key, label in (("GYRO", "gyro"), ("ACCL", "accel")):
            f = OUT / "imu" / f"{tag}_{key}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            T = d.t_video_s.to_numpy() - s0
            m = (T >= 0) & (T <= a.seconds)
            vals = d.iloc[:, 1:4].to_numpy()[m]
            for j, ax in enumerate(d.columns[1:4]):
                rr.send_columns(f"imu/{n}/{label}_{ax}" if label == "gyro" else f"imu_accel/{n}/{ax}",
                                indexes=[rr.TimeColumn("common_time", duration=T[m])],
                                columns=rr.Scalars.columns(scalars=vals[:, j]))
        print("logged", n, flush=True)

    # VO trajectory of HL
    hl = next(v for v in vids if v.stem.startswith("HL"))
    vo = pd.read_csv(OUT / "trajectory" / f"{hl.stem}_vo.csv")
    rep = json.loads((REPORTS / "trajectory.json").read_text())
    K = np.array(rep["K"])
    s0 = starts[hl.stem]
    vo["T"] = vo.timestamp_s - s0
    rr.log("world/path", rr.LineStrips3D([vo[["tx", "ty", "tz"]].to_numpy()], colors=[255, 200, 0], radii=0.3),
           static=True)
    for r in vo[(vo["T"] >= 0) & (vo["T"] <= a.seconds)].itertuples():
        rr.set_time("common_time", duration=r.T)
        rr.log("world/camera_HL", rr.Transform3D(translation=[r.tx, r.ty, r.tz],
                                                 quaternion=rr.Quaternion(xyzw=[r.qx, r.qy, r.qz, r.qw])))
        if r.Index % 3 == 0:
            rr.log("world/camera_HL", rr.Pinhole(image_from_camera=K, width=int(K[0, 2] * 2), height=int(K[1, 2] * 2),
                                                 image_plane_distance=3.0))
        rr.log("vo/inliers", rr.Scalars(r.inliers))
        rr.log("vo/tracked", rr.Scalars(r.tracked * 100))
    print("saved", OUT / "pipeline.rrd")


if __name__ == "__main__":
    main()
