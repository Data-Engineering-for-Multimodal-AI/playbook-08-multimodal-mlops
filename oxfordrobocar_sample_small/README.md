# Oxford RobotCar sample

A small extract of one drive from the **Oxford RobotCar Dataset**, recorded on 2014-12-12 in Oxford, UK.

- Source: <https://robotcar-dataset.robots.ox.ac.uk/>
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.
  This folder is **not** covered by the repository's MIT license. You may use it for non-commercial
  purposes with attribution, and share changes under the same license. For commercial use, contact
  the dataset's authors.
- Citation: W. Maddern, G. Pascoe, C. Linegar and P. Newman, "1 Year, 1000km: The Oxford RobotCar
  Dataset", *The International Journal of Robotics Research* 36(1), 2017.
- The camera images show public streets and may include people and vehicles. See the dataset's
  privacy page before you republish them.

Timestamps are UNIX time in microseconds (the file name of each sensor reading).

| Path | Contents | Used by |
|---|---|---|
| `gps/ins.csv` | 126,361 INS readings at 50 Hz, 10:45:53 to 11:28:00 UTC (42 minutes): position, velocity north/east/down, roll, pitch, yaw | the INS features, labels, training and the closed-loop replay |
| `gps/gps.csv` | 12,636 GPS fixes for the same period | not used |
| `stereo/centre/` | 49 PNG frames (1280×960, raw Bayer GBRG) from the forward stereo camera, 3.6 s at about 10:56:38 UTC | camera features (`src/ingest/video_ingest.py`, `src/features/video_features.py`) |
| `stereo/left/`, `stereo/right/` | the other two stereo cameras, same 49 timestamps | not used |
| `mono_left/`, `mono_right/`, `mono_rear/` | 40 PNG frames each from the side and rear cameras | not used |
| `lms_front/`, `lms_rear/` | 181 2D LiDAR scans each (`.bin`) | not used |
| `ldmrs/` | 45 3D LiDAR scans (`.bin`) | not used |
| `stereo.timestamps`, `mono_left.timestamps`, `mono_right.timestamps`, `mono_rear.timestamps`, `lms_front.timestamps`, `lms_rear.timestamps`, `ldmrs.timestamps` | the dataset's timestamp index for each sensor | not used |

The camera frames cover 3.6 seconds of the 42-minute INS log, so only 15 INS windows have a camera
frame within 0.1 s. That is why the model trains on the INS windows and the camera features are optional
(`model.use_video_features` in `config/params.yaml`).
