"""Task 5: monocular visual odometry (KLT tracks + 5-point essential matrix RANSAC) on Ego HL camera.

This is a ROUGH VISUAL-ODOMETRY estimate, not a SLAM reconstruction (no mapping, loop closure or bundle
adjustment). Translation is UP TO SCALE: per-frame unit translation direction from recoverPose, scaled by the
de-rotated median feature parallax (heuristic proxy assuming roughly constant scene depth). No metric claims.

Calibration: none supplied. Pinhole model, principal point at image centre, focal length from an ASSUMED
horizontal FOV (GoPro 'Linear' digital lens mode, distortion already removed in-camera -> zero distortion).
Validation: VO inter-frame rotation angles vs. the camera's own IMU orientation (GPMF CORI quaternion).

Usage: python src/visual_odometry.py [video] [--hfov 87] [--width 960]
Outputs: outputs/trajectory/<cam>_vo_tum.txt, <cam>_vo.csv, reports/trajectory.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from common import OUT, RAW, REPORTS

MIN_INLIERS = 40
MIN_TRACKS = 200
MIN_PARALLAX_PX = 0.5
KF_PARALLAX_PX = 8.0


def intrinsics(w, h, hfov_deg):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def run(path, hfov, width):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = width, int(round(H * width / W))
    K = intrinsics(w, h, hfov)
    Kinv = np.linalg.inv(K)
    R_wc, p_w = np.eye(3), np.zeros(3)          # current camera orientation / position (world = frame 0)
    kf_R, kf_p = np.eye(3), np.zeros(3)         # keyframe pose
    rows, prev, pts, kf_pts = [], None, None, None
    redetect, n_keyframes, lost_segments, in_loss = 0, 0, [], False
    t0 = time.perf_counter()
    fi = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        fi += 1
        g = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        rec = dict(frame_index=fi, timestamp_s=fi / fps, tracked=0, inliers=0, n_tracks=0, parallax_px=0.0,
                   rot_deg=np.nan, keyframe=0)
        if prev is None:
            rec["tracked"] = 1  # first frame defines the world origin
        elif pts is not None and len(pts) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, winSize=(21, 21), maxLevel=3)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pts - p0r, axis=2).ravel() < 1.0)
            a, b, kfa = pts[good].reshape(-1, 2), p1[good].reshape(-1, 2), kf_pts[good].reshape(-1, 2)
            rec["n_tracks"] = len(a)
            if len(a) >= 8:
                # (1) inter-frame ROTATION from the infinite homography H = K R K^-1 (valid for small baselines)
                Hm, hmask = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
                n_in = int(hmask.sum()) if hmask is not None else 0
                rec["inliers"] = n_in
                if Hm is not None and n_in >= MIN_INLIERS:
                    M = Kinv @ Hm @ K
                    U, _, Vt = np.linalg.svd(M)
                    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt   # nearest rotation (R_cur_prev)
                    R_wc = R_wc @ R.T
                    rec["tracked"] = 1
                    rec["rot_deg"] = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
                    # (2) TRANSLATION direction w.r.t. keyframe once de-rotated parallax is large enough
                    R_ck = R_wc.T @ kf_R                                    # keyframe->current rotation
                    ar = cv2.perspectiveTransform(kfa.reshape(-1, 1, 2), K @ R_ck @ Kinv).reshape(-1, 2)
                    par = float(np.median(np.linalg.norm(b - ar, axis=1)))
                    rec["parallax_px"] = par
                    if par > KF_PARALLAX_PX:
                        E, em = cv2.findEssentialMat(kfa, b, K, cv2.RANSAC, 0.999, 1.0)
                        if E is not None and E.shape == (3, 3):
                            _, Re, te, _, _ = cv2.recoverPose(E, kfa, b, K, distanceThresh=1e4, mask=em)
                            C_k = (-Re.T @ te).ravel()                      # cur centre in keyframe coords (unit)
                            p_w = kf_p + kf_R @ C_k * par                   # heuristic scale ~ parallax (arbitrary units)
                        kf_R, kf_p, kfa = R_wc.copy(), p_w.copy(), b.copy()
                        rec["keyframe"] = 1
                        n_keyframes += 1
            pts, kf_pts = b.reshape(-1, 1, 2), kfa.reshape(-1, 1, 2)
        if rec["tracked"] == 0:
            if not in_loss:
                lost_segments.append([fi, fi])
                in_loss = True
            lost_segments[-1][1] = fi
            kf_R, kf_p = R_wc.copy(), p_w.copy()   # re-initialise keyframe after loss
            pts = None
        else:
            in_loss = False
        if pts is None or len(pts) < MIN_TRACKS:
            new = cv2.goodFeaturesToTrack(g, maxCorners=1200, qualityLevel=0.01, minDistance=8)
            if new is not None:
                pts = new.astype(np.float32)          # restart tracks; current frame becomes keyframe
                kf_pts = pts.copy()
                kf_R, kf_p = R_wc.copy(), p_w.copy()
                redetect += 1
        if prev is None:
            kf_pts = pts.copy()
        q = Rotation.from_matrix(R_wc).as_quat()  # x y z w
        rec.update(tx=p_w[0], ty=p_w[1], tz=p_w[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3])
        rows.append(rec)
        prev = g
    elapsed = time.perf_counter() - t0
    return pd.DataFrame(rows), dict(fps=fps, K=K.tolist(), proc_res=f"{w}x{h}", src_res=f"{W}x{H}",
                                    redetections=redetect, keyframes=n_keyframes, lost_segments=lost_segments,
                                    processing_time_s=round(elapsed, 1))


def imu_rotation_check(df, cam_tag, k=5):
    """Compare |relative rotation| over k-frame windows: VO vs GoPro CORI orientation (axis-convention free)."""
    f = OUT / "imu" / f"{cam_tag}_CORI.csv"
    if not f.exists():
        return {"status": "no CORI data"}
    c = pd.read_csv(f)
    qc = c[[c.columns[2], c.columns[3], c.columns[4], c.columns[1]]].to_numpy()  # CORI is w,x,y,z -> x,y,z,w
    tc = c.t_video_s.to_numpy()
    # interpolate IMU orientation to frame times (nearest; CORI is ~1 sample per frame)
    idx = np.clip(np.searchsorted(tc, df.timestamp_s.to_numpy()), 0, len(tc) - 1)
    Rimu = Rotation.from_quat(qc[idx])
    Rvo = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy())
    valid = df.tracked.to_numpy().astype(bool)
    a_vo, a_imu = [], []
    for i in range(0, len(df) - k, k):
        if valid[i:i + k + 1].all():
            a_vo.append(np.degrees((Rvo[i].inv() * Rvo[i + k]).magnitude()))
            a_imu.append(np.degrees((Rimu[i].inv() * Rimu[i + k]).magnitude()))
    a_vo, a_imu = np.array(a_vo), np.array(a_imu)
    tot_vo = np.degrees((Rvo[0].inv() * Rvo[len(df) - 1]).magnitude())
    tot_imu = np.degrees((Rimu[0].inv() * Rimu[len(df) - 1]).magnitude())
    return {"window_frames": k, "n_windows": int(len(a_vo)),
            "pearson_r": round(float(np.corrcoef(a_vo, a_imu)[0, 1]), 3),
            "median_abs_err_deg": round(float(np.median(np.abs(a_vo - a_imu))), 3),
            "median_imu_angle_deg": round(float(np.median(a_imu)), 3),
            "ratio_vo_over_imu_median": round(float(np.median(a_vo[a_imu > 0.5] / a_imu[a_imu > 0.5])), 3),
            "start_to_end_rotation_deg": {"vo": round(float(tot_vo), 2), "imu_cori": round(float(tot_imu), 2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(RAW / "ego" / "HL_CH01_005_1013.MP4"))
    ap.add_argument("--hfov", type=float, default=87.0)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    path = Path(a.video)
    out = OUT / "trajectory"
    out.mkdir(exist_ok=True)
    df, info = run(path, a.hfov, a.width)
    df.to_csv(out / f"{path.stem}_vo.csv", index=False)
    with open(out / f"{path.stem}_vo_tum.txt", "w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw  (video media time, s; up-to-scale; tracked frames only)\n")
        for r in df[df.tracked == 1].itertuples():
            fh.write(f"{r.timestamp_s:.6f} {r.tx:.6f} {r.ty:.6f} {r.tz:.6f} {r.qx:.6f} {r.qy:.6f} {r.qz:.6f} {r.qw:.6f}\n")
    n = len(df)
    rep = {"method": "monocular VO: KLT + fwd/bwd check; "
                     "rotation per frame from infinite homography K^-1 H K (SVD-orthonormalised); translation direction at keyframes "
                     "(de-rotated parallax > 8px) from essential matrix vs keyframe, scaled by that parallax (heuristic, arbitrary units)",
           "classification": "rough visual odometry (NOT SLAM: no map, no loop closure, no BA)",
           "video": f"{path.parent.name}/{path.name}", "assumed_hfov_deg": a.hfov,
           "calibration": "assumed pinhole, cx,cy=centre, zero distortion (GoPro Linear lens mode); not calibrated",
           **info, "frames": n, "frames_tracked": int(df.tracked.sum()),
           "tracked_pct": round(100 * df.tracked.mean(), 2),
           "n_loss_segments": len(info["lost_segments"]),
           "frames_zero_translation_low_parallax": int(((df.parallax_px <= MIN_PARALLAX_PX) & (df.tracked == 1)).sum()),
           "median_inliers": float(df.inliers[df.tracked == 1].median()),
           "path_length_units": float(np.sum(np.linalg.norm(np.diff(df[["tx", "ty", "tz"]].to_numpy(), axis=0), axis=1))),
           "start_end_distance_units": float(np.linalg.norm(df[["tx", "ty", "tz"]].iloc[-1] - df[["tx", "ty", "tz"]].iloc[0])),
           "imu_rotation_validation": imu_rotation_check(df, f"{path.parent.name}_{path.stem}")}
    (REPORTS / "trajectory.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(df.tx, df.tz)
    ax[0].set_title("VO trajectory top view (x-z, arbitrary scale)")
    ax[0].axis("equal")
    ax[1].plot(df.timestamp_s, df.inliers, lw=0.7, label="inliers")
    ax[1].plot(df.timestamp_s, df.n_tracks, lw=0.7, label="tracks")
    ax[1].scatter(df.timestamp_s[df.tracked == 0], np.zeros((df.tracked == 0).sum()), c="r", s=6, label="lost")
    ax[1].legend()
    ax[1].set_title("tracking quality")
    ax[2].plot(df.timestamp_s, df.rot_deg, lw=0.7)
    ax[2].set_title("VO inter-frame rotation (deg)")
    plt.tight_layout()
    plt.savefig(out / f"{path.stem}_vo_plot.png", dpi=110)


if __name__ == "__main__":
    main()
