# ML readiness report

Abhay Krishna, ML Engineer Assessment (Jupiter AI Labs)

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
