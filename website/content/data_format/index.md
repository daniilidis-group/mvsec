---
date: 2017-08-08T21:07:13+01:00
title: Data Format
weight: 15
---
## ROS Bag Data Format
Each sequence consists of a data ROS bag, with the following topics:
<ul>
<li><b>/davis/left/events</b> (dvs_msgs/EventArray) - Events from the left DAVIS camera.</li>
<li><b>/davis/left/image_raw</b> (sensor_msgs/Image) - Grayscale images from the left DAVIS camera.</li>
<li><b>/davis/left/imu</b> (sensor_msgs/Imu) - IMU readings from the left DAVIS camera.</li>
<li><b>/davis/right/events</b> (dvs_msgs/EventArray) - Events from the right DAVIS camera.</li>
<li><b>/davis/right/image_raw</b> (sensor_msgs/Image) - Grayscale images from the right DAVIS camera.</li>
<li><b>/davis/right/imu</b> (sensor_msgs/Imu) - IMU readings from the right DAVIS camera.</li>
<li><b>/velodyne_point_cloud</b> (sensor_msgs/PointCloud2) - Point clouds from the Velodyne (one per spin).</li>
<li><b>/cam0/image_raw</b> (sensor_msgs/Image) - Grayscale images from the left VI-Sensor camera.</li>
<li><b>/cam1/image_raw</b> (sensor_msgs/Image) - Grayscale images from the right VI-Sensor camera.</li>
<li><b>/imu0</b> (sensor_msgs/Imu) - IMU readings from the VI-Sensor.</li>
<li><b>/cust_imu0</b> (visensor_node/visensor_imu) - Full IMU readings from the VI-Sensor (including magnetometer, temperature and pressure).</li>
<ul>

Two sets of custom messages are used, dvs_msgs/EventArray from <a href="https://github.com/uzh-rpg/rpg_dvs_ros">rpg_dvs_ros</a> and visensor_node/visensor_imu from <a href="https://github.com/ethz-asl/visensor_node">visensor_node</a>. The visensor_node package is optional if you do not need the extra IMU outputs (magnetometer, temperature and pressure.

In addition, each corresponding ground truth bag contains the following topics:
<ul>
<li><b>/davis/left/depth_image_raw</b> (sensor_msgs/Image) - Depth maps for the left DAVIS camera at a given timestamp (note, these images are saved using the CV_32FC1 format (i.e. floats).</li>
<li><b>/davis/left/blended_image_rect</b> (sensor_msgs/Image) - Visualization of all events from the left DAVIS that are 25ms from each left depth map superimposed on the depth map. This message gives a preview of what each sequence looks like.</li>
<li><b>/davis/left/odometry</b> (geometry_msgs/PoseStamped) - Pose output using <a href="https://www.ri.cmu.edu/publications/loam-lidar-odometry-and-mapping-in-real-time/">LOAM</a>. These poses are locally consistent, but may experience drift over time. Used to stitch point clouds together to generate depth maps.</li>
<li><b>/davis/left/pose</b> (geometry_msgs/PoseStamped) - Pose output using <a href="https://google-cartographer-ros.readthedocs.io/en/latest/">Google Cartographer</a>. These poses are globally loop closed, and can be assumed to have minimal drift. Note that these poses were optimized using Cartographer's 2D mapping, which does <b>not</b> optimize over the height (Z axis). Pitch and roll are still optimized, however.</li>
<li><b>/davis/right/depth_image_raw</b> (sensor_msgs/Image) - Depth maps for the right DAVIS camera at a given timestamp.</li>
<li><b>/davis/right/blended_image_rect</b> (sensor_msgs/Image) - Visualization of all events from the right DAVIS that are 25ms from each right depth map superimposed on the depth map. This message gives a preview of what each sequence looks like.</li>
</ul>

## Text Format

Coming soon!