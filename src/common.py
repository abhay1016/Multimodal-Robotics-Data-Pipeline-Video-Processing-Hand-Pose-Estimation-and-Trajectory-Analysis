"""Shared paths and helpers."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
for d in (OUT, REPORTS):
    d.mkdir(exist_ok=True)


_WINGET = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"


def _find(name):
    p = shutil.which(name)
    if p:
        return p
    hits = sorted(_WINGET.glob(f"Gyan.FFmpeg*/ffmpeg-*/bin/{name}.exe"))
    if hits:
        return str(hits[-1])
    raise FileNotFoundError(f"{name} not found; install FFmpeg (e.g. winget install Gyan.FFmpeg)")


def ffmpeg_bin():
    return _find("ffmpeg")


def ffprobe_bin():
    return _find("ffprobe")


def videos(group):
    """group: 'ego' or 'umi'. Sorted list of video paths."""
    exts = {".mp4", ".mov"}
    allv = sorted(p for p in RAW.rglob("*") if p.suffix.lower() in exts)
    if group == "umi":
        return [p for p in allv if "umi" in str(p).lower()]
    return [p for p in allv if "umi" not in str(p).lower()]
