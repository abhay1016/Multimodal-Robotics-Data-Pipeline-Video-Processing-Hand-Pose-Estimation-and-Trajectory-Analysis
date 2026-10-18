# Technical write-up

Abhay Krishna, ML Engineer Assessment (Jupiter AI Labs)

All numbers below come from the auto-generated files next to this one (`inspection.md`, `sync.json`, `hands.md`, `trajectory.json`, `imu_summary.json`, `lerobot_validation.txt`).

## 1. Inspection (Tasks 1 and 4)

* **Recording hardware.** 8 × GoPro (`firmware H24.01.02.10.00`, encoder "GoPro H.265"). Each file has 4 streams:
  * `hvc1` HEVC Main 1920×1080 `yuvj420p`, 29.97 fps (30000/1001)
  * `mp4a` AAC 48 kHz stereo
  * `tmcd` SMPTE drop-frame timecode
  * `gpmd` GoPro GPMF telemetry, one payload ≈ 1 s
* **Timing.** Nominal, average and PTS-derived fps are all 29.97. Frame-interval std is 0 ms, with 0 gaps and 0 duplicated PTS, so the files are constant frame rate. A full decode (`ffmpeg -f null`) reports no errors on any file.
* **Durations.** Ego 83.2 to 88.3 s, UMI 90.9 to 91.2 s. The differences come from cameras being started and stopped by hand, not from corruption.
* **Orientation.** Ego wrist cams (WL, WR) carry a `-180°` display-matrix rotation, i.e. they were mounted upside-down. OpenCV applies it automatically, so all processing sees upright frames.
* **Do Ego and UMI correspond?** No.
  * Ego take `CH01_005_1013`: creation 2026-08-20 05:17 UTC, timecode 10:47.
  * UMI take `CH01_005_1014`: creation 2026-09-14 09:32 UTC, timecode 14:32.
  * The scene content differs too. Ego shows test-tube handling at a desk, seen from head and wrist cameras. UMI shows handheld UMI-style grippers (GL/GR gripper cams with a strong fisheye) in a lab with ArUco markers, plus HL/HR head cams.
  * Within each group the 4 cameras record the same activity from different viewpoints.

### UMI-specific notes (Task 4)

* **Streams.** Same as Ego, including real IMU at ≈199 Hz, CORI/GRAV orientation, timecode and audio. GL/GR use a wide fisheye lens (visible barrel distortion), so they need a fisheye/Kannala-Brandt or omni model for SLAM. The pinhole model used for Ego HL would not be valid here.
* **Alignment within UMI.** Possible by the same audio GCC-PHAT method: `src/sync_ego.py` is camera-agnostic, and swapping `videos("ego")` for `videos("umi")` runs it. Timecode values are only 0.5 to 3 s apart, but for Ego we showed timecode disagrees with audio by up to 17 s, so timecode must be validated before it is trusted.
* **Alignment UMI ↔ Ego.** Not possible. They are different sessions recorded 25 days apart, with no shared event. We do not claim alignment.
* **Using UMI in a robotics dataset.**
  * The gripper camera is the policy's wrist observation (`observation.images.gripper_left/right`).
  * The camera trajectory from fisheye SLAM is the end-effector pose (with a fixed camera→TCP extrinsic).
  * **Gripper aperture** is missing. It is needed per frame, either from an encoder, or estimated from the fiducial markers on the gripper fingers seen in the gripper camera, as the UMI paper does. It must be expressed in mm or as a normalised 0 to 1 value.
  * The action is then `[Δpose(6/9D), gripper_width]` at the dataset fps.
* **Limitations.** There is no camera calibration, so fisheye intrinsics are unknown; a checkerboard or ArUco-board calibration per camera is needed. The ArUco marker size is unknown, so markers cannot fix metric scale yet. There is no gripper-width signal and no robot or teleop actions. Camera-to-gripper extrinsics are unknown.

## 2. Ego synchronization (Task 2)

**Evidence considered.**
1. Embedded timecode and creation_time.
2. Audio, since all 4 Ego cams record the same room.
3. GPMF timestamps (STMP). These are per-camera relative clocks, which also need a common reference.

**Rejected: timecode and creation_time.** HL `10:47:16;23` and HR `10:47:34;03` imply HR started 17.3 s after HL. Audio shows the two media timelines are only 36.8 ms apart. Visual inspection confirms audio: at the same frame index, HL and HR show the same hand pose (`outputs/frames/contact_sheet.jpg`). So the GoPro clocks were not mutually synced, and timecode is not valid sync evidence here.

**Method.** GCC-PHAT cross-correlation of the full mono 8 kHz audio tracks for all 6 pairs, with a ±120 s search window and parabolic sub-sample refinement (`src/sync_ego.py`).

Convention: `d_i` = t(cam i) − t(HL) for the same physical event.

| Camera | Offset vs HL | Frames @29.97 | GCC peak / 2nd peak |
|---|---|---|---|
| HR | **+36.8 ms** | +1.10 | 21.5 |
| WL | **−3169.4 ms** | −94.99 | 3.9 |
| WR | **−1966.6 ms** | −58.94 | 7.1 |

**Validation.**
* **Loop closure.** Offsets estimated independently per pair must satisfy `d_ik = d_ij + d_jk`. The errors are 1.87, 0.006, −0.72 and 1.15 ms, all ≤ 1.9 ms, which is well below one frame (33.4 ms).
* **Clock drift.** HR's offset estimated on the first third and on the last third of the clip is 36.776 vs 36.777 ms. There is no measurable drift over ~80 s. For WL/WR only the last third could be measured; it agrees with the full-clip estimate to 0.3 ms and 0.1 ms.
* **Visual check.** `outputs/ego_sync_2x2.mp4` plays the 4 cams on the common timeline; hand contacts line up across views.

**Common timeline.** T = 0 is when the last camera (HR) has started. The start times in each camera's media clock are HL 3.169 s, HR 3.206 s, WL 0.000 s and WR 1.203 s. The overlap is **82.08 s**.

**Residual uncertainty.**
* Audio-level precision is about ±2 ms, judging from loop closure.
* Frame-level alignment is limited by frame quantisation to ±16.7 ms, because cameras expose at independent phases with no genlock.
* The acoustic path difference between head and wrist mics is under 1 m, i.e. under 3 ms.
* Rolling shutter adds about ±10 ms within a frame.
* WL/WR peak ratios are lower (3.9 to 8.8), since the wrist mics are muffled by the hands. Loop closure still confirms them.

## 3. Hand keypoints (Task 3)

**Setup.** MediaPipe Hands 0.10.14 in video/tracking mode: `max_num_hands=2`, `model_complexity=1`, detection and tracking confidence 0.5. Frames are downscaled to 960×540 for speed. Landmarks are stored both normalised and in pixels at the original 1920×1080.

**Results** (60 s synced segment, 1798 frames per camera):

| Cam | ≥1 hand | Both hands | No detection | Low-conf (<0.7) hands | Label conflicts | Proc. fps |
|---|---|---|---|---|---|---|
| HL (head) | 1798 (100 %) | 1798 | 0 | 3 | 0 | 11.3 |
| HR (head) | 1798 (100 %) | 1798 | 0 | 0 | 0 | 12.3 |
| WL (wrist) | 1668 (92.8 %) | 751 | 130 | 343 | 120 | 11.1 |
| WR (wrist) | 137 (7.6 %) | 3 | 1661 | 17 | 3 | 15.4 |

Hardware: AMD Ryzen 7 7735HS, CPU only, about 2.5 min per camera minute. Four cameras ran in parallel processes.

**Handedness.** MediaPipe's handedness classifier assumes a *mirrored* selfie image. GoPro footage is not mirrored, so every label is **swapped** (`handedness`). The raw label is kept as `handedness_raw_mediapipe`.
* **Verification.** In frames where both hands were detected, the corrected "Right" wrist lies to the right of the "Left" wrist in **100 %** of head-cam frames (HL and HR, 1798 frames each) and in **99.2 %** of WL frames. This is the expected egocentric layout for head cams.
* **Wrist cams are the weak spot.** The view is close-up and partial, and WR often sees no full hand, so 7.6 % detection is a genuine limitation rather than a bug. When two hands receive the same label, both are flagged `label_conflict`. In the dataset only the higher-scoring one is kept.

**Missing and low-confidence detections.**
* Every processed frame is written. Frames without a hand have `status:"no_detection"`, `hands:[]` in the JSON and a row with `hand_slot=-1` in the CSV.
* **Nothing is interpolated.** In LeRobot, missing hands are zero-filled with `hand_mask=0`, so consumers must apply the mask.
* Low-confidence hands are kept but flagged with `low_confidence=true`, and their score is stored.
* If interpolation is wanted later, it should be a separate feature carrying an `interpolated` flag.

**Schema.** Documented inside every `*_hands.json`. Fields: frame_index, timestamp_s, t_common_s, status, num_hands, and `hands[]` with handedness, raw label, score, low_confidence, landmarks_norm (21×xyz) and landmarks_px (21×uv). Landmark order follows MediaPipe: 0 wrist, 1–4 thumb, 5–8 index, 9–12 middle, 13–16 ring, 17–20 pinky.

## 4. Trajectory (Task 5)

**Recording choice: Ego HL head cam.**
* It has the most static, textured background (desk, racks, markers), so hands occupy only a small part of the frame.
* It uses the GoPro *Linear* lens mode, where the camera removes distortion, so a pinhole model is defensible. The UMI gripper fisheye would need a fisheye model, and the wrist cams are dominated by the moving hand.
* It has the matching IMU orientation (CORI), which we use for validation.

**Calibration used.** No calibration was provided, so all values are assumed:
* Pinhole model, principal point at the image centre, zero distortion.
* fx = fy from an assumed **HFOV of 87°**, a typical GoPro Linear 16:9 value. At the 960×540 processing resolution this gives fx = 506 px.
* The IMU comparison shows a VO/IMU rotation ratio of 0.95, which is consistent with the focal length being roughly right.
* A proper calibration (checkerboard, or the GoPro lens metadata) is required for any accuracy claim.

**Method** (`src/visual_odometry.py`). Rough monocular **visual odometry**, explicitly **not SLAM**: there is no map, no loop closure and no bundle adjustment.
* KLT tracking with a forward-backward check (1 px).
* Per-frame **rotation** from the RANSAC infinite homography, R = K⁻¹HK, SVD-orthonormalised. This is valid because frame-to-frame translation is tiny compared with scene depth.
* At **keyframes**, when de-rotated parallax exceeds 8 px: **translation direction** from the 5-point essential matrix against the previous keyframe.
* Translation magnitude is the de-rotated parallax in pixels, a *heuristic* that assumes roughly constant depth. **The scale is arbitrary.** No metric claims are made.

**First attempt failed, and why.** Pure frame-to-frame essential matrices tracked only 0.8 % of frames: the near-zero baseline fails the cheirality test. Checking against the IMU caught this, since the end-to-end rotation was 2.6° VO vs 21.3° IMU. That motivated the homography plus keyframe design.

**Quality** (full video, 2633 frames):
* **100 % of frames tracked**, 0 tracking-loss segments, 3 feature re-detections, 54 translation keyframes, median 403 homography inliers.
* **IMU validation, axis-convention free.** Rotation magnitudes over 5-frame windows were compared with the GoPro CORI quaternion across 526 windows: Pearson **r = 0.79**, median absolute error **0.06°**, against a median IMU motion of 0.34° per window.
* **Drift.** Start-to-end rotation is 14.5° (VO) vs 21.3° (IMU), about **7° of rotational drift over 88 s**. Translation drift cannot be quantified: there is no ground truth and the camera returns to roughly its start pose, with start-to-end distance 5 units against a path length of 476 units.
* **Outputs.** `outputs/trajectory/HL_CH01_005_1013_vo_tum.txt` (TUM format, video media time), `_vo.csv` (per-frame inliers, tracks, parallax, keyframe and tracked flags), `_vo_plot.png`, and the Rerun 3D view.

**How this would improve.**
1. **Calibration.** Intrinsics and a fisheye/KB model fix the focal-length bias in rotation and translation direction.
2. **Visual-inertial SLAM.** ORB-SLAM3 or OpenVINS fed with the real 200 Hz GYRO/ACCL (already extracted) would give metric scale (gravity plus accelerometer), robustness to fast head motion and less rotation drift. This needs camera-IMU extrinsics and time-offset calibration (e.g. Kalibr). GoPro stamps both on one clock, which makes the time offset small and estimable.
3. **Stereo.** HL and HR form a head-mounted pair synced to 37 ms. With extrinsics they would give metric depth, but they are not hardware-synced, so rolling shutter and the ~1-frame offset would have to be modelled.
4. **Loop closure and bundle adjustment** from full SLAM would remove the long-term drift.

## 5. IMU investigation

* **Source.** The GPMF `gpmd` track inside each MP4. Our dependency-free parser (`src/extract_imu.py`) applies SCAL and ORIN and exports:
  * ACCL (m/s², ≈199 Hz) and GYRO (rad/s, ≈199 Hz), axis order **ZXY** per ORIN
  * CORI and IORI quaternions, one per video frame
  * GRAV vector, ISO, shutter and white balance
* **This is real measured data, not simulated.** A sample at rest reads |acc| ≈ 9.9 m/s², which matches gravity.
* **Timestamps.** Each ~1.001 s payload is mapped to its `gpmd` packet PTS on the video media clock, and its samples are spread uniformly across the payload. This approximation is accurate to roughly 1 sample (5 ms). Payload STMP (µs) is exported for refinement.
* **Cross-camera IMU** inherits the audio offsets from §2.
* **Integration so far.**
  1. Per-frame `observation.imu.*` (nearest ACCL+GYRO sample) and `observation.imu_orientation.*` in the LeRobot dataset.
  2. Rerun time series on the common timeline.
  3. Validation of the VO rotation.
* **Next step.** Feed the raw IMU into VI-SLAM, and use the gyro to correlate cameras without audio. Cross-correlating angular-rate magnitude between rigidly co-mounted cams (HL and HR on one head) is a second, independent sync check.

## 6. ML readiness (Task 8)

**What the dataset is.** Egocentric human-demonstration data:
* 4 synced views
* 21-point hand pose for both hands, with masks
* real 6-axis IMU and orientation per camera
* camera ego-motion (VO, up to scale)

**Uses.** Hand-centric imitation learning (human-to-robot retargeting), action recognition and segmentation, hand-object interaction perception, and pretraining visual encoders on manipulation views.

**Policy definition.**
* **Observations at time t:**
  * `observation.images.{ego_head_left, ego_head_right, ego_wrist_left, ego_wrist_right}`
  * `observation.hand_keypoints.*` with `hand_mask.*`
  * `observation.imu.*`
  * `observation.camera_pose.ego_head_left` with `camera_pose_valid`
* **Action at time t, for a human-to-robot retargeted policy** (not in the dataset; it must be derived or recorded):
  * per hand, wrist 6-DoF pose in a head- or world-fixed frame at t+1, minus pose at t
  * gripper open/close from thumb-index distance (landmarks 4–8)
  * optionally finger joint angles for dexterous hands
* **For UMI:** `action = [Δ end-effector pose, gripper width]`.
* **Timestamps.** `timestamp = k/30 s` on the common synced timeline. Source frame indices are kept in `observation.source_frame_index`.
* **Episodes.** One episode is one continuous task attempt (here, the whole synced take). With more data, episodes should be cut at task boundaries such as "pick tube" or "pour", using annotated or IMU/hand-motion-detected idle periods, never at fixed lengths.

**Using hand keypoints.**
* Supervision targets for 3D hand-pose or contact estimators.
* Hand-trajectory prediction as an intermediate action representation (affordance or "where will the hand go").
* Retargeting to robot hand joints or grippers.
* Masking hands out of the scene for SLAM.
* Phase segmentation from hand velocity.

**Using trajectory estimates.**
* Compensating head ego-motion, so hand motion can be expressed in a world frame rather than a moving head frame.
* Spatial grounding of objects across time.
* Motion statistics: speed, smoothness, idle detection.
* After VI-SLAM, metric scene reconstruction for sim-to-real.

**Leakage-free splits.** Split by **session, operator and scene**, never by frame or by window. Adjacent frames are nearly identical, and the 4 cameras of one take see the same event, so all cameras of a take must land in the same split. Hold out entire operators or sessions for test. The current data has one take, so it can only be used as a pipeline test, not for a train/val/test split.

**Metrics.**
* **Data quality:**
  * hand-detection rate and mask coverage per camera (HL 100 %, WR 7.6 %)
  * sync loop-closure error (≤ 1.9 ms)
  * VO-vs-IMU rotation correlation (r = 0.79) and tracked-frame % (100 %)
  * proportion of low-confidence or label-conflict hands
* **Training validation:**
  * per-episode action MSE on held-out sessions
  * hand-keypoint PCK@0.05 on a manually labelled subset
  * task success rate in closed-loop evaluation, the true metric for a policy

**Limitations without robot actions or IMU.**
* There is no `action` label, so behaviour cloning is impossible until actions are derived through retargeting or recorded through teleop. Retargeted human actions also have an embodiment gap in kinematics and grasp.
* Without IMU, monocular VO has no metric scale or gravity direction and drifts. This data does have IMU, but no camera-IMU calibration yet.
* There is no gripper-state or force/contact signal.
* There is no camera calibration.
* There is a single take.

## 7. LeRobot dataset (Task 6)

**Version and location.** `lerobot==0.3.3`, `CODEBASE_VERSION = "v2.1"`, written to `outputs/lerobot/jupiter_ego_ch01_005`. It was built with `LeRobotDataset.create`, `add_frame` and `save_episode`, then reloaded with `LeRobotDataset(...)`; the output is in `lerobot_validation.txt`.

**Frame rate.** The dataset uses one `fps = 30`. The sources run at 29.97 fps CFR, so each dataset frame takes the nearest source frame on the synced timeline. The error is ≤ 16.7 ms, and about 1 source frame per 1000 is duplicated. Separate datasets were rejected because the 4 cameras observe one event and are synced; joint access matters for multi-view learning.

**Excluded.** UMI (a separate session with no shared sync) and audio.

**Features.**
* 4 × video at 640×360.
* Per camera:
  * hand keypoints (126 = 2 hands × 21 × xyz, slot 0 = left, slot 1 = right)
  * `hand_mask` (2)
  * `hand_score` (2)
  * `imu` (6, ACCL ZXY + GYRO ZXY)
  * `imu_orientation` (4, CORI wxyz)
* For HL only: `camera_pose` (7, xyz + quaternion xyzw, arbitrary scale) and `camera_pose_valid` (1).
* `source_frame_index` (4).

**Episodes and task.** 1 episode = the 60 s (1800 frames) synced segment of take 1013, with the task string "Bimanual test-tube handling at a lab bench".

**No `action` feature.** We do not fabricate actions; see §6 for the mapping. Additional fields needed for a policy:
* `action` (retargeted wrist pose and gripper, or UMI EE pose and gripper width)
* `observation.state` (robot joint positions or gripper width)
* calibrated `observation.camera_pose` in metres from VI-SLAM
