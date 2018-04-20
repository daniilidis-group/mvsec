---
date: 2017-08-08T21:07:13+01:00
title: Change Log
weight: 30
---
2018/04/20
Updated IMU to camera extrinsics for the indoor flying set. A different driver was used on the hexacopter than during calibration, which rotated the IMU axes to be aligned with the camera. The extrinsics in the calibration have been modified to reflect this.

2018/02/01
Updated links in download page to reflect the previous name change to the outdoor bags.

2018/01/24  
Renamed bags in outdoor_day and outdoor_night to follow the convention of outdoor_day/outdoor_day1_gt.bag instead of outdoor_day/west_philly_day_1_gt.bag, etc.

2018/01/23  
Removed erroneous extra text from begining of outdoor_night camchain.

2018/01/19  
Fixed incorrect DAVIS left camera to IMU transform in the indoor flying sequence calibration.

2018/01/13  
Fixed direction of VI-Sensor axes in sensor rig image. They were rotated 90 degrees incorrectly before.

2017/12/19  
Fixed incorrect description in camera calibrations. T_cn_cm1 is the transform taking a point from the previous camera to the current camera, not the other way around.

2017/12/17  
Fixed VI-Sensor images. Both the left and right images had been mapped to /visensor/left/ before.

2017/12/10  
Added motorcycle data.








