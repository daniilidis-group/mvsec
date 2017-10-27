---
date: 2017-08-08T21:07:13+01:00
title: Calibration
weight: 10
---
## Calibration Parameters
Each camera was intrinsically calibrated using <a href="https://github.com/ethz-asl/kalibr">Kalibr</a>, with the DAVIS images calibrated using the equidistant distortion model, and the VI-Sensor images calibrated using the standard radtan distortion model. The two different distortion models is due to the slightly smaller focal length (more fisheye) lenses used on the DAVIS cameras compared to the stock VI-Sensor lenses.

To rectify the VI-Sensor images, you can use the standard OpenCV or ROS rectification functions.

To rectify the DAVIS images and events, you will need to use the <a href="https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye">OpenCV fisheye rectification functions</a>. This amounts to simply adding the fisheye namespace in front of the usual function (e.g. cv::fisheye::undistortPoints vs cv::undistortPoints). Note that the same cv::remap function works on both sets of images (no fisheye namespace needed). ROS does not currently support the equidistant distortion model. However, you can look at these pull requests: <a href="https://github.com/ros/common_msgs/pull/109">one</a>, <a href="https://github.com/ros-perception/vision_opencv/pull/184">two</a>, to the common_msgs and vision_opencv repos to find changes to the ROS image processing pipeline that allow for this model. Once these pull requests are merged in, this will no longer be necessary.

For convenience, the mapping between each pixel in the distorted image and the corresponding pixel in the rectified image is stored for each camera as $SEQUENCE\_(left/right)\_(x/y)_map.txt. For example, to rectify an event (or any point) (x, y) in the left DAVIS camera for outdoor_day:

x\_rect = outdoor\_day\_left\_x(y, x)</br>
y\_rect = outdoor\_day\_left\_y(y, x)

The extrinsics between the lidar and the left DAVIS camera are provided, as well as extrinsics between all cameras, as well as between each camera and its own IMU. In addition, the ground truth pose has been transformed into the left DAVIS camera frame.

All intrinsic and extrinsic calibrations are stored in <strong>yaml</strong> format, roughly following the calibration yaml files output from <a href="https://github.com/ethz-asl/kalibr">Kalibr</a>.

## Calibration File Format

Each scene (corresponding to a single day of recording) has its own calibration file. Each file consists of:
<ul>
  <li>T_cam0_lidar: The 4x4 transformation that takes a point from the Velodyne frame to the left DAVIS camera frame.</li>
  <li>For each camera (0-3):
  <ul>
    <li>Distortion model and coefficients</li>
    <li>Intrinsics</li>
    <li>Rectification matrix</li>
    <li>Projection matrix</li>
    <li>Resolution</li>
    <li>The ROS topic corresponding to this camera</li>
    <li>T_cam_imu: The 4x4 transformation that takes a point from this camera's IMU frame (where applicable) to this camera's camera frame.</li>
    <li>T_cn_cnm1: The 4x4 transformation that takes a point from this camera's camera frame to the previous camera's camera frame (e.g. cam2->cam1, cam1->cam0).</li>
  </ul>
  </li>
</ul>