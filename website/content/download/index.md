---
date: 2017-08-08T21:07:13+01:00
title: Download
weight: 25
---

## ROS Bags

To process the bag files, you will need the <a href="https://github.com/uzh-rpg/rpg_dvs_ros">rpg_dvs_ros</a> package to read the events (in particular dvs_msgs). You may also optionally install the <a href="https://github.com/ethz-asl/visensor_node">visensor_node</a> to have access to the /cust_imu0 topic, which includes the magnetometer, pressure and temperature outputs of the VI-Sensor.

Sequences will be added to this page on a rolling basis. We also plan to include videos for each sequence.

Note that these bags are large (up to 27G).

<div style='float:left;margin-left:5%'>
<table>
<col width="30%">
<col width="10%">
<col width="50%">
<tr><td>Scene</td><td>Calibration</td><td>Sequence</td><td>Map/Image</td></tr>
<tr>
<td>Indoor flying (Note: No VI-Sensor data is available for this scene).</td>
<td>
<a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying_calib.zip">Calibration</a>
</td>
<td>
Indoor Flying 1 <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying1_data.bag">Data (1.2G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying1_gt.bag">Ground truth (2.6G)</a>
</td>
</tr>
<tr>
<td>
</td>
<td>
</td>
<td>
Indoor Flying 2 <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying2_data.bag">Data (1.7G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying2_gt.bag">Ground truth (3.2G)</a>
</td>
</tr>
<tr>
<td>
</td>
<td>
</td>
<td>
Indoor Flying 3 <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying3_data.bag">Data (1.8G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying3_gt.bag">Ground truth (3.5G)</a>
</td>
</tr>
<tr>
<td>
</td>
<td>
</td>
<td>
Indoor Flying 4 <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying4_data.bag">Data (419M)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying4_gt.bag">Ground truth (738M)</a>
</td>
</tr>
<tr>
<td>Outdoor Driving Day (Note: A hardware failure caused the grayscale images on the right DAVIS grayscale images for this scene to be corrupted. However, VI-Sensor grayscale images are available).</td>
<td>
<a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day_calib.zip">Calibration</a>
</td>
<td>
Outdoor Day 1 <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_data.bag">Data (9.7G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_gt.bag">Ground truth (9.5G)</a>
</td>
<td>
<a target="_blank" href="../figs/gt_maps/west_philly_day1_traj.jpg">
<img src="../figs/gt_maps/west_philly_day1_traj.jpg" alt="outdoor_day1" style="max-height:150px"/>
</a>
</td>

</tr>
<tr>
<td>
</td>
<td>
</td>
<td>
Outdoor Day 2 <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day2_data.bag">Data (27G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day2_gt.bag">Ground truth (23G)</a>
</td>
<td>
<a target="_blank" href="../figs/gt_maps/west_philly_day2_traj.jpg">
<img src="../figs/gt_maps/west_philly_day2_traj.jpg" alt="outdoor_day2" style="max-height:150px"/>
</a>
</td>
</tr>
<tr>
<td>Outdoor Driving Night</td>
<td>
<a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night_calib.zip">Calibration</a>
</td>
<td>
Outdoor Night 1 <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night1_data.bag">Data (8.1G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night1_gt.bag">Ground truth (9.5G)</a>
</td>
<td>
<a target="_blank" href="../figs/gt_maps/west_philly_night1_traj.jpg">
<img src="../figs/gt_maps/west_philly_night1_traj.jpg" alt="outdoor_night1" style="max-height:150px"/>
</a>
</td>
</tr>
<tr>
<td>
</td>
<td>
</td>
<td>
Outdoor Night 2 <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night2_data.bag">Data (11G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night2_gt.bag">Ground truth (11G)</a>
</td>
<td>
<a target="_blank" href="../figs/gt_maps/west_philly_night2_traj.jpg">
<img src="../figs/gt_maps/west_philly_night2_traj.jpg" alt="outdoor_night2" style="max-height:150px"/>
</a>
</td>
</tr>
<tr>
<td>
</td>
<td>
</td>
<td>
Outdoor Night 3 <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night3_data.bag">Data (9G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night3_gt.bag">Ground truth (11G)</a>
</td>
<td>
<a target="_blank" href="../figs/gt_maps/west_philly_night3_traj.jpg">
<img src="../figs/gt_maps/west_philly_night3_traj.jpg" alt="outdoor_night3" style="max-height:150px"/>
</a>
</td>
</tr>
<tr>
<td>Motorcycle (Note: No lidar).</td>
<td>
<a href="http://visiondata.cis.upenn.edu/mvsec/motorcycle/motorcycle_calib.zip">Calibration</a>
</td>
<td>
Highway 1 <a href="http://visiondata.cis.upenn.edu/mvsec/motorcycle/motorcycle_data.bag">Data (42G)</a> <a href="http://visiondata.cis.upenn.edu/mvsec/motorcycle/motorcycle_gt.bag">Ground truth (659K)</a>
</td>
</tr>
</table>
</div>

<BR CLEAR="all">
## Text Files

Coming soon!
