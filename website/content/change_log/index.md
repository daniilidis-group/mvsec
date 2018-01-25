---
date: 2017-08-08T21:07:13+01:00
title: Change Log
weight: 30
---
Dec 10 2017 - Added motorcycle data.

Dec 17 2017 - Fixed VI-Sensor images. Both the left and right images had been mapped to /visensor/left/ before.

Dec 19 2017 - Fixed incorrect description in camera calibrations. T_cn_cm1 is the transform taking a point from the previous camera to the current camera, not the other way around.

Jan 13 2018 - Fixed direction of VI-Sensor axes in sensor rig image. They were rotated 90 degrees incorrectly before.

Jan 19 2018 - Fixed incorrect DAVIS left camera to IMU transform in the indoor flying sequence calibration.

Jan 23 2018 - Removed erroneous extra text from begining of outdoor_night camchain.

Jan 24 2018 - Renamed bags in outdoor_day and outdoor_night to follow the convention of outdoor_day/outdoor_day1_gt.bag instead of outdoor_day/west_philly_day_1_gt.bag, etc.