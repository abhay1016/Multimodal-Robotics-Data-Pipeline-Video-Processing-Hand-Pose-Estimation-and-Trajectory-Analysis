"""Task 1 & 4: inspect all videos (all container streams, timing, quality) -> reports/inspection.{md,csv,json}."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import ffmpeg_bin

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports"
from common import ffprobe_bin
FFPROBE = ffprobe_bin()


def probe(path):
    cmd = [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def video_pts(path):
    """Per-packet presentation timestamps of the first video stream (seconds)."""
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
           "-of", "csv=p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.split()
    pts = []
    for line in out:
        t = line.split(",")[0]
        if t not in ("", "N/A"):
            pts.append(float(t))
    return np.sort(np.array(pts))


def decode_check(path):
    """Full decode; any error output means corruption."""
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stderr.strip()[:300]


def ratio(s):
    try:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else float("nan")
    except Exception:
        return float("nan")


def main(full_decode=True):
    OUT.mkdir(exist_ok=True)
    files = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in (".mp4", ".mov"))
    rows, streams_rows, raw = [], [], {}
    for f in files:
        p = probe(f)
        raw[f"{f.parent.name}/{f.name}"] = p
        fmt = p["format"]
        v = next(s for s in p["streams"] if s["codec_type"] == "video")
        pts = video_pts(f)
        dt = np.diff(pts) if len(pts) > 1 else np.array([np.nan])
        nominal = ratio(v.get("r_frame_rate", "0/1"))
        avg = ratio(v.get("avg_frame_rate", "0/1"))
        med_dt = float(np.median(dt))
        gaps = int(np.sum(dt > 1.5 * med_dt))
        dups = int(np.sum(dt < 0.5 * med_dt))
        for s in p["streams"]:
            streams_rows.append({
                "file": f"{f.parent.name}/{f.name}", "index": s["index"], "type": s.get("codec_type"),
                "codec": s.get("codec_name"), "tag": s.get("codec_tag_string"),
                "handler": s.get("tags", {}).get("handler_name"),
                "timecode": s.get("tags", {}).get("timecode"),
                "duration_s": s.get("duration"), "nb_frames": s.get("nb_frames"),
                "sample_rate": s.get("sample_rate"), "channels": s.get("channels"),
            })
        rows.append({
            "file": f"{f.parent.name}/{f.name}",
            "group": "UMI" if "umi" in str(f.parent).lower() else "Ego",
            "size_MB": round(int(fmt["size"]) / 1e6, 1),
            "container_dur_s": round(float(fmt.get("duration", "nan")), 3),
            "video_dur_s": round(float(v.get("duration", "nan")), 3),
            "codec": v.get("codec_name"), "profile": v.get("profile"), "pix_fmt": v.get("pix_fmt"),
            "res": f"{v['width']}x{v['height']}",
            "rotation": next((sd.get("rotation") for sd in v.get("side_data_list", []) if "rotation" in sd), None),
            "fps_nominal": round(nominal, 3), "fps_avg": round(avg, 3),
            "fps_from_pts": round(1 / med_dt, 3) if med_dt else None,
            "nb_frames_meta": v.get("nb_frames"), "n_pts": len(pts),
            "dt_min_ms": round(float(np.min(dt)) * 1e3, 3), "dt_max_ms": round(float(np.max(dt)) * 1e3, 3),
            "dt_std_ms": round(float(np.std(dt)) * 1e3, 3),
            "VFR": bool(np.std(dt) > 0.05 * med_dt),
            "gaps(>1.5dt)": gaps, "dup(<0.5dt)": dups,
            "first_pts_s": round(float(pts[0]), 4) if len(pts) else None,
            "n_streams": len(p["streams"]),
            "stream_types": ",".join(f"{s.get('codec_type')}:{s.get('codec_tag_string') or s.get('codec_name')}" for s in p["streams"]),
            "creation_time": fmt.get("tags", {}).get("creation_time"),
            "encoder/make": fmt.get("tags", {}).get("encoder") or fmt.get("tags", {}).get("com.apple.quicktime.make")
                            or fmt.get("tags", {}).get("major_brand"),
            "timecode": next((s.get("tags", {}).get("timecode") for s in p["streams"] if s.get("tags", {}).get("timecode")), None),
            "decode_errors": decode_check(f) if full_decode else "skipped",
        })
        print("inspected", f.name, flush=True)
    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(streams_rows)
    df.to_csv(OUT / "inspection.csv", index=False)
    sdf.to_csv(OUT / "inspection_streams.csv", index=False)
    (OUT / "inspection_raw_ffprobe.json").write_text(json.dumps(raw, indent=1))
    with open(OUT / "inspection.md", "w", encoding="utf-8") as fh:
        fh.write("# Video inspection report (auto-generated by src/inspect_videos.py)\n\n## Per-file summary\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n\n## All container streams\n\n")
        fh.write(sdf.to_markdown(index=False))
        fh.write("\n")
    print(df.to_string())
    print(sdf.to_string())


if __name__ == "__main__":
    main(full_decode="--fast" not in sys.argv)
