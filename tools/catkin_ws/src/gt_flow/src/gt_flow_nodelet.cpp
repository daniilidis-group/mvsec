/* -*-c++-*--------------------------------------------------------------------
 * 2017 Bernd Pfrommer bernd.pfrommer@gmail.com
 */

#include "gt_flow/gt_flow_nodelet.h"
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <boost/range/irange.hpp>

namespace gt_flow {
  using boost::irange;
  void GTFlowNodelet::onInit() {
    ros::NodeHandle nh = getPrivateNodeHandle();
    image_transport::ImageTransport it(nh);
    imageSub_.reset(new image_transport::SubscriberFilter());
    imageSub_->subscribe(it, "depth_image_rect", 1);
    poseSub_.reset(new Subscriber<PoseStampedMsg>(nh, "pose", 1));
    
    cameraInfoSub_ = nh.subscribe("camera_info", 1,
                                  &GTFlowNodelet::callbackCameraInfo, this);
    if (useApproxSync_) {
      approxSync_.reset(new ApproxTimeSynchronizer2(
                          ApproxSyncPolicy2(10/*q size*/), *imageSub_, *poseSub_));
      approxSync_->registerCallback(&GTFlowNodelet::callbackDepthAndPose, this);
    } else {
      sync_.reset(new ExactSynchronizer2(*imageSub_, *poseSub_, 2));
      sync_->registerCallback(&GTFlowNodelet::callbackDepthAndPose, this);
    }
  }

  void GTFlowNodelet::callbackCameraInfo(CameraInfoConstPtr const &camInfoMsg) {
    cameraModel_.fromCameraInfo(camInfoMsg);
    ROS_INFO("got camera info!");
    undistortedPoints_.resize(camInfoMsg->height * camInfoMsg->width);
    for (unsigned int v = 0; v < camInfoMsg->height; v++) {
      for (unsigned int u = 0; u < camInfoMsg->width; u++) {
        cv::Point2d uv_rect = cameraModel_.rectifyPoint(cv::Point2d(u,v));
        undistortedPoints_[v * camInfoMsg->width + u] = cameraModel_.projectPixelTo3dRay(uv_rect);
      }
    }
    cameraInfoSub_.shutdown();
  }

  static Eigen::Isometry3d pose_msg_to_eigen(const geometry_msgs::Pose &pose) {
    Eigen::Quaterniond orientation;
    Eigen::Vector3d translation;
    Eigen::Isometry3d pe;
    pe.matrix() = Eigen::Isometry3d::MatrixType::Identity();
    tf::pointMsgToEigen(pose.position, translation);
    tf::quaternionMsgToEigen(pose.orientation, orientation);
    pe.linear() = orientation.toRotationMatrix();
    pe.translation() = translation;
    return (pe);
  }

  void GTFlowNodelet::callbackDepthAndPose(ImageConstPtr const &depthMsg,
                                           PoseStampedConstPtr const &poseMsg) {
    ROS_INFO("got depth/pose callback");
    if (!cameraModel_.initialized()) {
      ROS_WARN("camera model not initialized, dropping message!");
      return;
    }
    cv::Size res = cameraModel_.fullResolution();
    if (depthMsg->width != (unsigned int) res.width ||
        depthMsg->height != (unsigned int) res.height) {
      ROS_ERROR("resolution mismatch between caminfo and depth image!");
      return;
    }
        
    Eigen::Isometry3d T_c_w = pose_msg_to_eigen(poseMsg->pose);
    //std::cout << T_c_w.matrix() << std::endl;
    const auto &t = poseMsg->header.stamp;
    if (!hasInitPose_) {
      initPose_ = T_c_w;
      lastTime_ = poseMsg->header.stamp;
      hasInitPose_ = true;
      return;
    }
    const double dtinv = 1.0/std::min((t-lastTime_).toSec(), 1e-6);
    const auto dR = initPose_.inverse() * T_c_w;
    
    std::cout << "dR: " << dR.matrix() << std::endl;
    const Eigen::AngleAxisd aa(dR.linear() * dtinv);
    const Eigen::Vector3d omega = aa.axis() * aa.angle();
    const Eigen::Vector3d v    = dR.translation() * dtinv;
    std::cout << "omega: " << omega.transpose() << std::endl;
    std::cout << "v: " << v.transpose() << std::endl;
    cv_bridge::CvImageConstPtr const cv_ptr = cv_bridge::toCvShare(
      depthMsg, sensor_msgs::image_encodings::TYPE_32FC1);
    
    cv::Mat const img = cv_ptr->image;

     for (const auto v : irange(0u, depthMsg->height)) {
      for (const auto u : irange(0u, depthMsg->width)) {
        double Z = img.at<float>(v, u);
        cv::Point3d X  = undistortedPoints_[v * depthMsg->width + u] * Z;
      }
    }
  }
}  // namespace
