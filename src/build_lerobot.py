"""Task 6: build a LeRobot v2.1 dataset (lerobot==0.3.3) from the synchronized Ego streams.

Design
- fps = 30 (LeRobot needs one integer fps). Source cams are 29.97 fps CFR; every dataset frame k at common time
  T_k = k/30 takes, per camera, the NEAREST source frame round((start_i + T_k) * fps_i) (error <= 16.7 ms,
  ~1 duplicated source frame per 1000). Source frame indices are stored for traceability.
- One episode = the synchronized segment of take CH01_005_1013 (one continuous demonstration; we do not cut it
  at arbitrary times because no task-phase labels exist).
- UMI take (1014) is EXCLUDED: different session, no shared sync evidence with Ego (see reports).
- Real, measured signals only: GoPro IMU (ACCL+GYRO, nearest sample), hand keypoints (MediaPipe detections,
  zero-filled + mask=0 when absent, never interpolated), VO pose (arbitrary scale, valid flag).
- No `action` feature: no robot/teleop actions exist and we do not fabricate them.

Usage: python src/build_lerobot.py [--seconds 60] [--res 640x360]
"""
import argparse
import json
import shutil

import cv2
import numpy as np
import pandas as pd

from common import OUT, REPORTS, videos

FPS = 30
CAM_NAMES = {"HL": "ego_head_left", "HR": "ego_head_right", "WL": "ego_wrist_left", "WR": "ego_wrist_right"}
HAND_NAMES = [f"{side}_lm{k}_{ax}" for side in ("left", "right") for k in range(21) for ax in "xyz"]


def load_hands(stem):
    f = OUT / "hands" / f"{stem}_hands.json"
    if not f.exists():
        return {}
    return {fr["frame_index"]: fr for fr in json.loads(f.read_text())["frames"]}


def hand_vec(fr):
    """-> (126,) [left 21x3, right 21x3] normalized, (2,) mask, (2,) score. Missing hand -> zeros + mask 0."""
    v, m, s = np.zeros((2, 21, 3), np.float32), np.zeros(2, np.float32), np.zeros(2, np.float32)
    for h in (fr or {}).get("hands", []):
        slot = 0 if h["handedness"] == "Left" else 1
        if m[slot] and s[slot] >= h["handedness_score"]:
            continue  # duplicate label: keep higher-scoring hand
        v[slot] = np.array(h["landmarks_norm"], np.float32)
        m[slot], s[slot] = 1.0, h["handedness_score"]
    return v.reshape(-1), m, s


def nearest(tv, vals, t):
    """Nearest-in-time sample (tv sorted)."""
    i = int(np.clip(np.searchsorted(tv, t), 1, len(tv) - 1))
    return vals[i - 1] if (t - tv[i - 1]) < (tv[i] - t) else vals[i]


def main():
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--res", default="640x360")
    a = ap.parse_args()
    W, H = map(int, a.res.split("x"))
    sync = json.loads((REPORTS / "sync.json").read_text())
    starts = sync["common_timeline"]["start_in_each_cam_s"]
    dur = min(a.seconds, sync["common_timeline"]["overlap_duration_s"]) if a.seconds > 0 else \
        sync["common_timeline"]["overlap_duration_s"]
    n_frames = int(np.floor(dur * FPS))
    vids = videos("ego")
    keys = {v.stem: CAM_NAMES[v.stem[:2]] for v in vids}

    features = {}
    for v in vids:
        c = keys[v.stem]
        features[f"observation.images.{c}"] = {"dtype": "video", "shape": (H, W, 3),
                                               "names": ["height", "width", "channels"]}
        features[f"observation.hand_keypoints.{c}"] = {"dtype": "float32", "shape": (126,), "names": HAND_NAMES}
        features[f"observation.hand_mask.{c}"] = {"dtype": "float32", "shape": (2,), "names": ["left", "right"]}
        features[f"observation.hand_score.{c}"] = {"dtype": "float32", "shape": (2,), "names": ["left", "right"]}
        features[f"observation.imu.{c}"] = {"dtype": "float32", "shape": (6,),
                                            "names": ["acc_z", "acc_x", "acc_y", "gyro_z", "gyro_x", "gyro_y"]}
        features[f"observation.imu_orientation.{c}"] = {"dtype": "float32", "shape": (4,),
                                                        "names": ["w", "x", "y", "z"]}
    features["observation.camera_pose.ego_head_left"] = {"dtype": "float32", "shape": (7,),
                                                         "names": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]}
    features["observation.camera_pose_valid.ego_head_left"] = {"dtype": "float32", "shape": (1,), "names": ["valid"]}
    features["observation.source_frame_index"] = {"dtype": "int64", "shape": (len(vids),),
                                                  "names": [keys[v.stem] for v in vids]}

    root = OUT / "lerobot" / "jupiter_ego_ch01_005"
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id="local/jupiter_ego_ch01_005", fps=FPS, features=features, root=root,
                               robot_type="human_egocentric_gopro", use_videos=True,
                               image_writer_threads=12)

    caps, fpss, cur, last = {}, {}, {}, {}
    hands, imu, cori = {}, {}, {}
    for v in vids:
        caps[v.stem] = cv2.VideoCapture(str(v))
        fpss[v.stem] = caps[v.stem].get(cv2.CAP_PROP_FPS)
        cur[v.stem] = -1
        hands[v.stem] = load_hands(v.stem)
        tag = f"{v.parent.name}_{v.stem}"
        acc = pd.read_csv(OUT / "imu" / f"{tag}_ACCL.csv")
        gyr = pd.read_csv(OUT / "imu" / f"{tag}_GYRO.csv")
        imu[v.stem] = (acc.t_video_s.to_numpy(), acc.iloc[:, 1:4].to_numpy(), gyr.t_video_s.to_numpy(),
                       gyr.iloc[:, 1:4].to_numpy())
        co = pd.read_csv(OUT / "imu" / f"{tag}_CORI.csv")
        cori[v.stem] = (co.t_video_s.to_numpy(), co.iloc[:, 1:5].to_numpy())
    vo = pd.read_csv(OUT / "trajectory" / "HL_CH01_005_1013_vo.csv").set_index("frame_index")
    hl = next(v.stem for v in vids if v.stem.startswith("HL"))

    task = "Bimanual test-tube handling at a lab bench (egocentric human demonstration)"
    for k in range(n_frames):
        T = k / FPS
        frame = {}
        src_idx = []
        for v in vids:
            s = v.stem
            fi = int(round((starts[s] + T) * fpss[s]))
            while cur[s] < fi:  # sequential decode, nearest-frame resampling
                ok, img = caps[s].read()
                cur[s] += 1
                if ok:
                    last[s] = img
            src_idx.append(fi)
            c = keys[s]
            frame[f"observation.images.{c}"] = cv2.cvtColor(cv2.resize(last[s], (W, H), interpolation=cv2.INTER_AREA),
                                                            cv2.COLOR_BGR2RGB)
            hv, hm, hs = hand_vec(hands[s].get(fi))
            frame[f"observation.hand_keypoints.{c}"] = hv
            frame[f"observation.hand_mask.{c}"] = hm
            frame[f"observation.hand_score.{c}"] = hs
            t_media = fi / fpss[s]
            ta, va, tg, vg = imu[s]
            frame[f"observation.imu.{c}"] = np.concatenate([nearest(ta, va, t_media),
                                                            nearest(tg, vg, t_media)]).astype(np.float32)
            frame[f"observation.imu_orientation.{c}"] = nearest(*cori[s], t_media).astype(np.float32)
            if s == hl:
                r = vo.loc[fi] if fi in vo.index else None
                ok = r is not None and r.tracked == 1
                frame["observation.camera_pose.ego_head_left"] = (
                    r[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(np.float32) if ok else np.zeros(7, np.float32))
                frame["observation.camera_pose_valid.ego_head_left"] = np.array([1.0 if ok else 0.0], np.float32)
        frame["observation.source_frame_index"] = np.array(src_idx, np.int64)
        ds.add_frame(frame, task=task)
        if k % 300 == 0:
            print(f"frame {k}/{n_frames}", flush=True)
    ds.save_episode()
    print("saved; CODEBASE_VERSION =", CODEBASE_VERSION)

    # ---- validation: reload through the official API ----
    ds2 = LeRobotDataset("local/jupiter_ego_ch01_005", root=root)
    lines = [f"codebase_version (info.json): {ds2.meta.info['codebase_version']}",
             f"num_episodes: {ds2.num_episodes}", f"num_frames: {ds2.num_frames}", f"fps: {ds2.fps}",
             "features:"] + [f"  {k}: {v['dtype']} {tuple(v['shape'])}" for k, v in ds2.features.items()]
    item = ds2[len(ds2) // 2]
    lines.append(f"sample frame (index {len(ds2) // 2}):")
    for k, v in item.items():
        lines.append(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else ''} "
                     f"{v.dtype if hasattr(v, 'dtype') else type(v).__name__} "
                     f"{'' if hasattr(v, 'shape') and v.numel() > 8 else v}")
    txt = "\n".join(lines)
    (REPORTS / "lerobot_validation.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
