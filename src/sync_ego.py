"""Task 2: Ego camera synchronization by audio GCC-PHAT cross-correlation.

Outputs: reports/sync.json, reports/sync.md, outputs/ego_sync_2x2.mp4
Offset convention: d_i = (time in cam i) - (time in reference cam) of the same physical event.
Common timeline T maps to cam-i media time  t_i = T + d_i - T0shift.
"""
import itertools
import json
import subprocess
import sys

import cv2
import numpy as np

from common import OUT, REPORTS, ffmpeg_bin, videos

FS = 8000
MAX_LAG_S = 120.0


def load_audio(path, fs=FS):
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(fs),
           "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    a = np.frombuffer(r.stdout, dtype=np.float32).astype(np.float64)
    return a - a.mean()


def gcc_phat(x, y, fs=FS, max_lag_s=MAX_LAG_S):
    """Return (lag_s, peak_ratio): y is delayed by lag_s relative to x."""
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    X, Y = np.fft.rfft(x, n), np.fft.rfft(y, n)
    R = Y * np.conj(X)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    m = int(max_lag_s * fs)
    cc = np.concatenate([cc[-m:], cc[:m + 1]])  # lags -m..m
    k = int(np.argmax(np.abs(cc)))
    peak = np.abs(cc[k])
    mask = np.ones_like(cc, bool)
    mask[max(0, k - fs // 20): k + fs // 20] = False  # exclude ±50 ms around peak
    second = np.abs(cc[mask]).max()
    # parabolic sub-sample refinement
    if 0 < k < len(cc) - 1:
        a, b, c = np.abs(cc[k - 1]), peak, np.abs(cc[k + 1])
        frac = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        frac = 0.0
    return (k - m + frac) / fs, float(peak / (second + 1e-12))


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    m = dict(fps=cap.get(cv2.CAP_PROP_FPS), n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
             w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    m["dur"] = m["n"] / m["fps"]
    return m


def main(tile_seconds=60.0):
    vids = videos("ego")
    names = [v.stem for v in vids]
    metas = [video_meta(v) for v in vids]
    audio = [load_audio(v) for v in vids]
    res = {"method": "GCC-PHAT audio cross-correlation, 8 kHz mono, sub-sample parabolic refinement",
           "reference": names[0], "cameras": {}, "pairwise": {}, "validation": {}}
    if any(a is None for a in audio):
        res["status"] = "FAILED: missing audio track(s) - " + str([n for n, a in zip(names, audio) if a is None])
        (REPORTS / "sync.json").write_text(json.dumps(res, indent=2))
        print(res)
        return

    # pairwise offsets on full audio
    pair = {}
    for i, j in itertools.combinations(range(len(vids)), 2):
        lag, pr = gcc_phat(audio[i], audio[j])
        pair[(i, j)] = lag
        res["pairwise"][f"{names[i]}->{names[j]}"] = {"offset_s": round(lag, 5), "peak_ratio": round(pr, 2)}

    # loop-closure consistency: d_ik should equal d_ij + d_jk
    loops = []
    for i, j, k in itertools.combinations(range(len(vids)), 3):
        err = pair[(i, k)] - (pair[(i, j)] + pair[(j, k)])
        loops.append({"triplet": f"{names[i]},{names[j]},{names[k]}", "closure_err_ms": round(err * 1e3, 3)})
    res["validation"]["triplet_closure"] = loops

    # drift check: offset estimated separately on first and last third of the overlap (vs reference)
    drift = {}
    for j in range(1, len(vids)):
        d = pair[(0, j)]
        segs = []
        L = int(min(len(audio[0]), len(audio[j])) / 3)
        for part in (0, 2):
            s0 = part * L
            x = audio[0][s0:s0 + L]
            s1 = int(s0 + d * FS)
            if s1 < 0 or s1 + L > len(audio[j]):
                segs.append(None)
                continue
            y = audio[j][s1:s1 + L]
            lag, pr = gcc_phat(x, y, max_lag_s=2.0)
            segs.append(round((d + lag) * 1e3, 3))
        drift[names[j]] = {"offset_first_third_ms": segs[0], "offset_last_third_ms": segs[1]}
    res["validation"]["drift_first_vs_last_third"] = drift

    for idx, (n, m) in enumerate(zip(names, metas)):
        d = 0.0 if idx == 0 else pair[(0, idx)]
        res["cameras"][n] = {"offset_vs_ref_ms": round(d * 1e3, 3), "offset_vs_ref_frames": round(d * m["fps"], 2),
                             "fps": m["fps"], "frames": m["n"], "duration_s": round(m["dur"], 3)}

    # common timeline: T=0 when all cams have started; t_i = T + start_i
    d = np.array([0.0] + [pair[(0, j)] for j in range(1, len(vids))])
    T0 = max(-d)            # ref time at which every cam has begun... t_i = t_ref + d_i >= 0
    starts = T0 + d         # media time in each cam at common T=0
    overlap = min(m["dur"] - s for m, s in zip(metas, starts))
    res["common_timeline"] = {"start_in_each_cam_s": {n: round(float(s), 4) for n, s in zip(names, starts)},
                              "overlap_duration_s": round(float(overlap), 3)}
    res["status"] = "OK"
    (REPORTS / "sync.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # 2x2 tiled synchronized video of the first `tile_seconds` of overlap
    dur = min(tile_seconds, overlap)
    cmd = [ffmpeg_bin(), "-y", "-v", "error"]
    for v, s in zip(vids, starts):
        cmd += ["-ss", f"{s:.4f}", "-t", f"{dur:.3f}", "-i", str(v)]
    labels = "".join(
        f"[{i}:v]scale=640:360,fps=30,drawtext=text='{names[i]} start={starts[i]:.3f}s':x=10:y=10:fontsize=20:fontcolor=yellow:box=1:boxcolor=black[v{i}];"
        for i in range(len(vids)))
    n = len(vids)
    if n == 4:
        fc = labels + "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
    else:
        fc = labels + "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "26", "-c:a", "aac", str(OUT / "ego_sync_2x2.mp4")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:  # drawtext may be unavailable (no fontconfig) -> retry without labels
        cmd[cmd.index("-filter_complex") + 1] = \
            "".join(f"[{i}:v]scale=640:360,fps=30[v{i}];" for i in range(n)) + fc.split(";")[-1]
        r = subprocess.run(cmd, capture_output=True, text=True)
    print("tiled video:", "OK" if r.returncode == 0 else r.stderr[-500:])


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
