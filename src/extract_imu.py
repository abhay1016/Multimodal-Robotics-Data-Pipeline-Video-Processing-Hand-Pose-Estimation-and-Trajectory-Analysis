"""IMU investigation: extract REAL accelerometer/gyroscope (and other) telemetry from the GoPro GPMF
('gpmd' data track) embedded in every MP4. Dependency-free KLV parser (GPMF spec: github.com/gopro/gpmf-parser).

Per stream we apply SCAL, keep ORIN (axis order), and timestamp samples by spreading each ~1 s payload's
samples uniformly over that payload's [pts, pts+duration) on the *video media clock* (documented approximation;
STMP microsecond stamps are also exported when present).

Outputs: outputs/imu/<video>_<KEY>.csv, reports/imu_summary.json
"""
import json
import struct
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from common import OUT, REPORTS, ffmpeg_bin, ffprobe_bin, videos

FFPROBE = ffprobe_bin()
TYPES = {"b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I", "f": "f", "d": "d", "j": "q", "J": "Q"}
SENSOR_KEYS = {"ACCL", "GYRO", "GRAV", "CORI", "IORI", "MAGN", "GPS5", "GPS9", "SHUT", "ISOE", "WBAL"}


def klv_iter(buf, off=0, end=None):
    end = len(buf) if end is None else end
    while off + 8 <= end:
        key = buf[off:off + 4].decode("latin1")
        typ = chr(buf[off + 4])
        size = buf[off + 5]
        rep = struct.unpack(">H", buf[off + 6:off + 8])[0]
        n = size * rep
        data_off = off + 8
        yield key, typ, size, rep, data_off, n
        off = data_off + ((n + 3) & ~3)


def decode(buf, typ, size, rep, off, n):
    if typ == "c":
        return buf[off:off + n].decode("latin1", "ignore").rstrip("\x00")
    if typ == "U":
        return buf[off:off + n].decode("latin1")
    if typ not in TYPES:
        return None
    fmt = TYPES[typ]
    w = struct.calcsize(fmt)
    cnt = size // w
    arr = np.frombuffer(buf[off:off + n], dtype=">" + {"b": "i1", "B": "u1", "h": "i2", "H": "u2", "i": "i4", "I": "u4",
                                                         "f": "f4", "d": "f8", "q": "i8", "Q": "u8"}[fmt])
    return arr.reshape(rep, cnt).astype(np.float64) if cnt > 1 else arr.astype(np.float64)


def parse_payload(buf):
    """Return {KEY: dict(values, stmp, orin, name, units)} for one GPMF payload."""
    found = {}
    for key, typ, size, rep, off, n in klv_iter(buf):
        if key != "DEVC":
            continue
        for k2, t2, s2, r2, o2, n2 in klv_iter(buf, off, off + n):
            if k2 != "STRM":
                continue
            meta, scal = {}, None
            for k3, t3, s3, r3, o3, n3 in klv_iter(buf, o2, o2 + n2):
                if k3 == "SCAL":
                    scal = decode(buf, t3, s3, r3, o3, n3)
                elif k3 in ("STMP", "TSMP"):
                    meta[k3] = decode(buf, t3, s3, r3, o3, n3)
                elif k3 in ("ORIN", "STNM", "SIUN", "UNIT"):
                    meta[k3] = decode(buf, t3, s3, r3, o3, n3)
                elif k3 in SENSOR_KEYS or k3 == "GPSU":
                    v = decode(buf, t3, s3, r3, o3, n3)
                    if v is None:
                        continue
                    if scal is not None and not isinstance(v, str):
                        sc = np.atleast_1d(scal).astype(np.float64)
                        v = v / (sc if (v.ndim == 2 and sc.size == v.shape[1]) else sc[0])
                    found[k3] = dict(values=v, **meta)
    return found


def gpmd_packets(path):
    info = json.loads(subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
                                     capture_output=True, text=True).stdout)
    idx = next((s["index"] for s in info["streams"] if s.get("codec_tag_string") == "gpmd"), None)
    if idx is None:
        return None, []
    pk = json.loads(subprocess.run([FFPROBE, "-v", "error", "-select_streams", str(idx), "-show_packets",
                                    "-show_entries", "packet=pts_time,duration_time,size", "-of", "json", str(path)],
                                   capture_output=True, text=True).stdout)["packets"]
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", str(path), "-map", f"0:{idx}", "-c", "copy",
                          "-f", "data", "-"], capture_output=True).stdout
    out, off = [], 0
    for p in pk:
        sz = int(p["size"])
        out.append((float(p["pts_time"]), float(p.get("duration_time", 1.0)), raw[off:off + sz]))
        off += sz
    return idx, out


def main():
    imu_dir = OUT / "imu"
    imu_dir.mkdir(exist_ok=True)
    summary = {}
    for v in videos("ego") + videos("umi"):
        tag = f"{v.parent.name}_{v.stem}"
        idx, packets = gpmd_packets(v)
        if idx is None:
            summary[tag] = "no gpmd telemetry track"
            continue
        series = defaultdict(list)
        meta = {}
        for pts, dur, buf in packets:
            for key, d in parse_payload(buf).items():
                vals = d["values"]
                if isinstance(vals, str):
                    series[key].append(pd.DataFrame({"t_video_s": [pts], "value": [vals]}))
                    continue
                vals = np.atleast_2d(vals.T).T if vals.ndim == 1 else vals
                n = len(vals)
                t = pts + dur * np.arange(n) / n
                df = pd.DataFrame(vals, columns=[f"c{i}" for i in range(vals.shape[1])])
                df.insert(0, "t_video_s", t)
                if "STMP" in d:
                    df["payload_stmp_us"] = float(np.atleast_1d(d["STMP"])[0])
                series[key].append(df)
                meta.setdefault(key, {k: (str(x) if not isinstance(x, (str, float, int)) else x)
                                      for k, x in d.items() if k in ("ORIN", "STNM", "SIUN", "UNIT")})
        s = {}
        for key, dfs in series.items():
            df = pd.concat(dfs, ignore_index=True)
            orin = meta.get(key, {}).get("ORIN")
            if isinstance(orin, str) and len(orin) == df.shape[1] - 1 - ("payload_stmp_us" in df):
                df = df.rename(columns={f"c{i}": f"{a}" for i, a in enumerate(orin)})
            df.to_csv(imu_dir / f"{tag}_{key}.csv", index=False)
            entry = {"samples": len(df), **meta.get(key, {})}
            if len(df) > 1 and key not in ("GPSU",):
                span = df.t_video_s.iloc[-1] - df.t_video_s.iloc[0]
                entry["rate_hz"] = round((len(df) - 1) / span, 2) if span > 0 else None
            if key == "GPSU":
                entry["first_utc"] = str(df.value.iloc[0])
            s[key] = entry
        summary[tag] = {"gpmd_stream_index": idx, "payloads": len(packets), "streams": s}
        print(tag, json.dumps(s), flush=True)
    (REPORTS / "imu_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
