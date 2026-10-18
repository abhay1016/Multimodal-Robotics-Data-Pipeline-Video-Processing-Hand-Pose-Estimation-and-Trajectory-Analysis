# Multimodal Robotics Data Pipeline: Jupiter AI Labs ML Engineer Assessment

**Candidate:** Abhay Krishna  
**Role:** ML Engineer – Robotics

The pipeline runs end to end: container inspection, real GoPro IMU extraction, audio-based Ego sync, 21-point hand keypoints, monocular VO validated against the on-board IMU, a LeRobot **v2.1** dataset that reloads through the official API, and a Rerun recording.
Everything reported is either **measured** or **explicitly labelled as estimated, assumed or heuristic**.

## 0. Key findings

| Topic | Result |
|---|---|
| Hardware that recorded the data | 8 × **GoPro** (firmware `H24.01.02.10.00`, HERO13-class). HEVC 1920×1080, **29.97 fps CFR**, AAC 48 kHz stereo, `tmcd` timecode, **`gpmd` GPMF telemetry**. |
| Data quality | All 8 files decode with no errors. 0 PTS gaps, 0 duplicate PTS, frame-interval std 0 ms (true CFR). Wrist cams carry a `-180°` display-rotation flag (they were mounted upside-down; OpenCV auto-applies it). |
| **IMU** | **Real IMU exists in every file**: ACCL (m/s²) + GYRO (rad/s) at **≈199 Hz**, axis order `ZXY`. There is also CORI (camera-orientation quaternion, 1 per frame), IORI, GRAV, exposure/ISO and GPS9. These are extracted with our own GPMF parser (`src/extract_imu.py`). |
| Ego vs UMI | **They are different sessions.** Ego = take `1013`, 2026-08-20 05:17 UTC, test-tube handling at a desk. UMI = take `1014`, 2026-09-14 09:32 UTC, gripper manipulation with ArUco markers. They are not corresponding recordings and cannot be aligned with each other. |
| Ego sync | **Audio GCC-PHAT**: offsets vs HL are **HR +36.8 ms (+1.10 fr)**, **WL −3169.4 ms (−94.99 fr)** and **WR −1966.6 ms (−58.94 fr)**. Triplet loop-closure error ≤ **1.9 ms**. HR offset drift between the first and last third of the clip is **0.001 ms**. The common overlap is **82.08 s**. Embedded timecode / creation_time **contradict** the audio by up to ~17 s, so the camera clocks are **not** mutually synced and timecode was rejected as sync evidence. |
| Hands | MediaPipe Hands, 60 s synced segment × 4 cams. Handedness is swapped for non-mirrored footage, and the swap is verified spatially. See §3. |
| Trajectory | **Rough monocular VO, not SLAM.** 100 % of 2633 frames tracked, 0 losses. Rotation is validated against the GoPro IMU orientation: per-window rotation-magnitude **r = 0.79**, median error **0.06°**, VO/IMU ratio 0.95. End-to-end rotation is 14.5° (VO) vs 21.3° (IMU), i.e. ~7° of drift over 88 s. Translation is **up to scale** (arbitrary units). |
| LeRobot | `lerobot==0.3.3` → `codebase_version: v2.1`. 1 episode, 1800 frames (60 s synced segment) @ 30 fps, 4 video features plus hands, IMU, pose and masks. Reloaded and printed in `reports/lerobot_validation.txt`. |

## 1. Setup

* Windows 11, AMD Ryzen 7 7735HS (8C/16T), 15 GB RAM, RTX 3050 Laptop 4 GB. **All processing ran on the CPU**; the GPU was unused.
* Python **3.10** (via `uv`). Two environments keep MediaPipe's pins away from torch/lerobot:
  * `.venv-cv`: `mediapipe==0.10.14`, `opencv-contrib-python`, `rerun-sdk==0.24.1`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tabulate`. Used for inspection, IMU, sync, hands, VO and Rerun.
  * `.venv`: `lerobot==0.3.3` (pulls `torch==2.7.1`, `datasets==3.6.0`, `av==17.1.0`), plus `opencv-python`, `pandas`. Used for the dataset.
  * Exact versions: `requirements-cv.txt`, `requirements-lerobot.txt` (from `uv pip freeze`).
* FFmpeg/ffprobe 9.0 (`winget install Gyan.FFmpeg`). `src/common.py` finds it on PATH or in the WinGet folder.

```bash
uv venv --python 3.10 .venv-cv && uv pip install --python .venv-cv -r requirements-cv.txt
uv venv --python 3.10 .venv    && uv pip install --python .venv    -r requirements-lerobot.txt
# raw videos -> data/raw/ego/*.MP4, data/raw/umi/*.MP4 (NOT included in the submission)
```

## 2. Run (in order)

```bash
.venv-cv/Scripts/python src/inspect_videos.py          # Task 1/4 -> reports/inspection.{md,csv}, inspection_streams.csv, raw ffprobe json
.venv-cv/Scripts/python src/extract_imu.py             # IMU     -> outputs/imu/*.csv, reports/imu_summary.json
.venv-cv/Scripts/python src/sync_ego.py 60             # Task 2  -> reports/sync.json, outputs/ego_sync_2x2.mp4
.venv-cv/Scripts/python src/hand_keypoints.py --seconds 60   # Task 3 (or --cams HR in parallel, then --report-only)
.venv-cv/Scripts/python src/visual_odometry.py         # Task 5  -> outputs/trajectory/*, reports/trajectory.json
.venv/Scripts/python    src/build_lerobot.py --seconds 60    # Task 6 -> outputs/lerobot/jupiter_ego_ch01_005 + reports/lerobot_validation.txt
.venv-cv/Scripts/python src/visualize_rerun.py --seconds 60  # Task 7 -> outputs/pipeline.rrd   (view: rerun outputs/pipeline.rrd)
```
Every compute-heavy step takes `--seconds 0` (or equivalent) to run on full videos. VO already runs on the full HL video.

**Segment choice.** Hands and Rerun use the first **60 s of the common synchronized timeline**, starting at T = 0, the moment all four Ego cameras are recording. The segment contains continuous bimanual manipulation. It was chosen to fit the time budget, not cherry-picked. The LeRobot dataset uses the same 60 s (1800 frames).

## 3. Deliverables (in the order of Section 6 of the brief)

| # | Deliverable | Location |
|---|---|---|
| 1 | Python source for every stage | `src/`: `inspect_videos.py`, `extract_imu.py`, `sync_ego.py`, `hand_keypoints.py`, `visual_odometry.py`, `build_lerobot.py`, `visualize_rerun.py`, `common.py` |
| 2 | LeRobot v2.1 dataset (loads via the official API) | `outputs/lerobot/jupiter_ego_ch01_005/`; regenerate with `src/build_lerobot.py`; load check in `reports/lerobot_validation.txt` |
| 3 | Synchronized Ego video | `outputs/ego_sync_2x2.mp4`, still `outputs/screenshots/ego_sync_2x2_t30s.jpg`; limitations in `reports/WRITEUP.md` §2 |
| 4 | Hand-keypoint annotations | `outputs/hands/*_hands.json` (schema embedded), `outputs/hands/*_hands.csv` |
| 5 | Annotated Ego previews | `outputs/hands/*_hands_preview.mp4`, `outputs/screenshots/hands_preview_samples.jpg` |
| 6 | Trajectory files and visualization | `outputs/trajectory/HL_CH01_005_1013_vo_tum.txt`, `_vo.csv`, `_vo_plot.png`; 3D view in the Rerun recording |
| 7 | Reports | Inspection/data quality: `reports/inspection.md` (+ `.csv`, `inspection_streams.csv`); sync: `reports/sync.md`, `sync.json`; hands: `reports/hands.md`; trajectory: `reports/trajectory.json` + `WRITEUP.md` §4; IMU: `reports/imu_summary.json` + `WRITEUP.md` §5; UMI: `WRITEUP.md` §1 |
| 8 | Rerun recording and screenshots | `outputs/pipeline.rrd` (open with `rerun outputs/pipeline.rrd`), `outputs/screenshots/rerun_viewer.png` |
| 9 | README | this file |
| 10 | ML readiness report | `reports/ml_readiness.md` |

Extra: real IMU CSVs extracted from the GoPro telemetry are in `outputs/imu/`, and a contact sheet of all 8 videos is in `outputs/screenshots/contact_sheet.jpg`. The raw videos are **not** included.


## 4. Confidentiality

Raw videos are not committed (`data/` is in `.gitignore`). The dataset videos are re-encoded, low-resolution derivatives kept locally. Delete `data/raw` after review.
