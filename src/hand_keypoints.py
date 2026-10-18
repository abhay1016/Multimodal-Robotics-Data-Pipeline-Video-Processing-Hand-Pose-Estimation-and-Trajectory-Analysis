"""Task 3: 21-point hand-keypoint annotation of Ego videos with MediaPipe Hands (legacy solutions API).

Usage: python src/hand_keypoints.py [--seconds 60] [--cams HL,HR,WL,WR]
The segment starts at the common synchronized timeline T=0 (reports/sync.json) so all cameras cover the
same physical interval; pass --seconds 0 for the full overlap.

Handedness: MediaPipe's classifier assumes a MIRRORED (selfie) image. GoPro egocentric footage is NOT mirrored,
so the raw label is swapped (Left<->Right). Both raw and corrected labels are stored.
Missing detections: never interpolated; frame is recorded with status="no_detection" and hands=[].
"""
import argparse
import json
import platform
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from common import OUT, REPORTS, videos

LOW_CONF = 0.7  # handedness score below this is flagged low_confidence (still stored)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
SWAP = {"Left": "Right", "Right": "Left"}


def process(path, start_s, seconds, out_dir, proc_width=960):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(start_s * fps))
    f1 = n_total if seconds <= 0 else min(n_total, f0 + int(round(seconds * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    scale = proc_width / W
    pw, ph = proc_width, int(round(H * scale))
    writer = cv2.VideoWriter(str(out_dir / f"{path.stem}_hands_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (pw, ph))
    frames, rows = [], []
    t0 = time.perf_counter()
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for fi in range(f0, f1):
            ok, bgr = cap.read()
            ts = fi / fps
            if not ok:
                frames.append({"frame_index": fi, "timestamp_s": round(ts, 6), "t_common_s": round(ts - start_s, 6),
                               "status": "decode_error", "num_hands": 0, "hands": []})
                continue
            small = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
            res = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            hs = []
            if res.multi_hand_landmarks:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    c = hd.classification[0]
                    norm = [[p.x, p.y, p.z] for p in lm.landmark]
                    assert len(norm) == 21
                    hs.append({"handedness": SWAP[c.label], "handedness_raw_mediapipe": c.label,
                               "handedness_score": round(c.score, 4), "low_confidence": c.score < LOW_CONF,
                               "landmarks_norm": [[round(v, 5) for v in p] for p in norm],
                               "landmarks_px": [[round(p[0] * W, 1), round(p[1] * H, 1)] for p in norm]})
                    mp_draw.draw_landmarks(small, lm, mp_hands.HAND_CONNECTIONS)
                    wx, wy = int(norm[0][0] * pw), int(norm[0][1] * ph)
                    cv2.putText(small, f"{SWAP[c.label]} {c.score:.2f}", (wx, wy + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)
                # two hands with the same corrected label -> keep both but flag the conflict
                if len(hs) == 2 and hs[0]["handedness"] == hs[1]["handedness"]:
                    for h in hs:
                        h["label_conflict"] = True
            status = "detected" if hs else "no_detection"
            cv2.putText(small, f"{path.stem} f={fi} t={ts:.3f}s {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if hs else (0, 0, 255), 2)
            writer.write(small)
            frames.append({"frame_index": fi, "timestamp_s": round(ts, 6), "t_common_s": round(ts - start_s, 6),
                           "status": status, "num_hands": len(hs), "hands": hs})
            for hi, h in enumerate(hs):
                row = {"camera": path.stem, "frame_index": fi, "timestamp_s": ts, "hand_slot": hi,
                       "handedness": h["handedness"], "handedness_raw": h["handedness_raw_mediapipe"],
                       "score": h["handedness_score"]}
                for k, (p, q) in enumerate(zip(h["landmarks_norm"], h["landmarks_px"])):
                    row.update({f"lm{k}_x": p[0], f"lm{k}_y": p[1], f"lm{k}_z": p[2], f"lm{k}_px": q[0],
                                f"lm{k}_py": q[1]})
                rows.append(row)
            if not hs:
                rows.append({"camera": path.stem, "frame_index": fi, "timestamp_s": ts, "hand_slot": -1,
                             "handedness": None})
    elapsed = time.perf_counter() - t0
    cap.release()
    writer.release()

    n = len(frames)
    det = sum(f["num_hands"] >= 1 for f in frames)
    both = sum(f["num_hands"] == 2 for f in frames)
    lefts = sum(any(h["handedness"] == "Left" for h in f["hands"]) for f in frames)
    rights = sum(any(h["handedness"] == "Right" for h in f["hands"]) for f in frames)
    allh = [h for f in frames for h in f["hands"]]
    # handedness sanity check: in non-mirrored egocentric view the RIGHT hand's wrist is usually right of
    # the LEFT hand's wrist when both are visible.
    order_ok = order_n = 0
    for f in frames:
        if f["num_hands"] == 2 and not f["hands"][0].get("label_conflict"):
            r = next(h for h in f["hands"] if h["handedness"] == "Right")
            l = next(h for h in f["hands"] if h["handedness"] == "Left")
            order_n += 1
            order_ok += r["landmarks_norm"][0][0] > l["landmarks_norm"][0][0]
    stats = {
        "camera": path.stem, "fps": fps, "resolution": f"{W}x{H}", "processed_at": f"{pw}x{ph}",
        "frame_range": [f0, f1 - 1], "segment_media_time_s": [round(f0 / fps, 3), round((f1 - 1) / fps, 3)],
        "frames_processed": n, "frames_ge1_hand": det, "frames_both_hands": both,
        "frames_no_detection": sum(f["status"] == "no_detection" for f in frames),
        "decode_errors": sum(f["status"] == "decode_error" for f in frames),
        "frames_with_left": lefts, "frames_with_right": rights,
        "hands_low_confidence": sum(h["low_confidence"] for h in allh),
        "frames_label_conflict": sum(1 for f in frames if f["hands"] and f["hands"][0].get("label_conflict")),
        "handedness_spatial_check": {"two_hand_frames": order_n,
                                     "right_wrist_right_of_left_pct": round(100 * order_ok / order_n, 1) if order_n else None},
        "processing_time_s": round(elapsed, 2), "proc_fps": round(n / elapsed, 2),
    }
    doc = {"schema": {
        "frame_index": "0-based index in the source video", "timestamp_s": "frame_index / fps (CFR verified in inspection)",
        "t_common_s": "time on the synchronized Ego timeline (sync.json)",
        "status": "detected | no_detection | decode_error", "hands[].handedness": "corrected for non-mirrored footage",
        "hands[].handedness_raw_mediapipe": "MediaPipe label (assumes mirrored input)",
        "hands[].landmarks_norm": "21 x [x,y,z]; x,y in [0,1] of image width/height, z relative depth (wrist origin)",
        "hands[].landmarks_px": "21 x [u,v] pixels at original resolution",
        "landmark_order": "MediaPipe: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky",
        "interpolation": "none - missing frames are explicit nulls"},
        "stats": stats, "frames": frames}
    (out_dir / f"{path.stem}_hands.json").write_text(json.dumps(doc))
    pd.DataFrame(rows).to_csv(out_dir / f"{path.stem}_hands.csv", index=False)
    print(json.dumps(stats), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--cams", default="")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    sync = json.loads((REPORTS / "sync.json").read_text())
    starts = sync["common_timeline"]["start_in_each_cam_s"]
    out_dir = OUT / "hands"
    out_dir.mkdir(exist_ok=True)
    allstats = []
    for v in videos("ego"):
        if a.report_only:  # aggregate stats of per-camera runs (e.g. run in parallel with --cams)
            f = out_dir / f"{v.stem}_hands.json"
            if f.exists():
                allstats.append(json.loads(f.read_text())["stats"])
            continue
        if a.cams and not any(v.stem.startswith(c) for c in a.cams.split(",")):
            continue
        allstats.append(process(v, starts[v.stem], a.seconds, out_dir))
    hw = f"{platform.processor()} | {platform.platform()} | MediaPipe {mp.__version__} CPU"
    df = pd.DataFrame([{k: v for k, v in s.items() if not isinstance(v, dict)} |
                       {"LR_spatial_check_%": s["handedness_spatial_check"]["right_wrist_right_of_left_pct"]}
                       for s in allstats])
    (REPORTS / "hands.md").write_text("# Hand keypoint annotation stats (auto-generated)\n\nHardware: " + hw +
                                      "\n\n" + df.to_markdown(index=False) + "\n")
    (REPORTS / "hands_stats.json").write_text(json.dumps({"hardware": hw, "cameras": allstats}, indent=2))
    print(df.to_string())


if __name__ == "__main__":
    main()
