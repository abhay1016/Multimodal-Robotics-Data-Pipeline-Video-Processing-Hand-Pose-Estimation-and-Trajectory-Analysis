# Ego synchronization report (generated from reports/sync.json)

Method: GCC-PHAT audio cross-correlation, 8 kHz mono, sub-sample parabolic refinement

| camera           |   offset_vs_ref_ms |   offset_vs_ref_frames |   fps |   frames |   duration_s |
|:-----------------|-------------------:|-----------------------:|------:|---------:|-------------:|
| HL_CH01_005_1013 |              0     |                   0    | 29.97 |     2633 |       87.854 |
| HR_CH01_005_1013 |             36.763 |                   1.1  | 29.97 |     2645 |       88.255 |
| WL_CH01_005_1013 |          -3169.36  |                 -94.99 | 29.97 |     2492 |       83.15  |
| WR_CH01_005_1013 |          -1966.64  |                 -58.94 | 29.97 |     2496 |       83.283 |

## Pairwise GCC-PHAT

| pair                               |   offset_s |   peak_ratio |
|:-----------------------------------|-----------:|-------------:|
| HL_CH01_005_1013->HR_CH01_005_1013 |    0.03676 |        21.47 |
| HL_CH01_005_1013->WL_CH01_005_1013 |   -3.16937 |         3.94 |
| HL_CH01_005_1013->WR_CH01_005_1013 |   -1.96664 |         7.08 |
| HR_CH01_005_1013->WL_CH01_005_1013 |   -3.208   |         3.9  |
| HR_CH01_005_1013->WR_CH01_005_1013 |   -2.00341 |         5.32 |
| WL_CH01_005_1013->WR_CH01_005_1013 |    1.20344 |         8.77 |

## Validation: triplet loop closure

| triplet                                            |   closure_err_ms |
|:---------------------------------------------------|-----------------:|
| HL_CH01_005_1013,HR_CH01_005_1013,WL_CH01_005_1013 |            1.871 |
| HL_CH01_005_1013,HR_CH01_005_1013,WR_CH01_005_1013 |            0.006 |
| HL_CH01_005_1013,WL_CH01_005_1013,WR_CH01_005_1013 |           -0.718 |
| HR_CH01_005_1013,WL_CH01_005_1013,WR_CH01_005_1013 |            1.147 |

## Validation: drift (offset on first vs last third)

| camera           |   offset_first_third_ms |   offset_last_third_ms |
|:-----------------|------------------------:|-----------------------:|
| HR_CH01_005_1013 |                  36.776 |                 36.777 |
| WL_CH01_005_1013 |                 nan     |              -3169.09  |
| WR_CH01_005_1013 |                 nan     |              -1966.51  |

## Timecode vs audio (timecode rejected: disagrees by seconds)

| camera           | timecode    | creation_time               |   offset_implied_by_timecode_ms |   offset_measured_audio_ms |
|:-----------------|:------------|:----------------------------|--------------------------------:|---------------------------:|
| HL_CH01_005_1013 | 10:47:16;23 | 2026-08-20T05:17:16.000000Z |                            -0   |                      0     |
| HR_CH01_005_1013 | 10:47:34;03 | 2026-08-20T05:17:34.000000Z |                        -17332.7 |                     36.763 |
| WL_CH01_005_1013 | 10:47:18;04 | 2026-08-20T05:17:18.000000Z |                         -1366   |                  -3169.36  |
| WR_CH01_005_1013 | 10:47:16;26 | 2026-08-20T05:17:16.000000Z |                          -100.1 |                  -1966.64  |

## Common timeline

{
 "start_in_each_cam_s": {
  "HL_CH01_005_1013": 3.1694,
  "HR_CH01_005_1013": 3.2061,
  "WL_CH01_005_1013": 0.0,
  "WR_CH01_005_1013": 1.2027
 },
 "overlap_duration_s": 82.08
}
