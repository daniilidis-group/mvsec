/* -*-c++-*--------------------------------------------------------------------
 * 2017 Bernd Pfrommer bernd.pfrommer@gmail.com
 */

#ifndef GT_FLOW_GT_FLOW_NODELET_H
#define GT_FLOW_GT_FLOW_NODELET_H

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <geometry_msgs/PoseStamped.h>
#include <image_geometry/pinhole_camera_model.h>

#include <ros/ros.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <nodelet/nodelet.h>

#include <opencv2/core.hpp>
#include <memory>
#include <vector>


namespace gt_flow {
  using sensor_msgs::ImageConstPtr;
  using ImageMsg = sensor_msgs::Image;
  using sensor_msgs::CameraInfo;
  using sensor_msgs::CameraInfoConstPtr;
  
  using PoseStampedMsg = geometry_msgs::PoseStamped;
  using geometry_msgs::PoseStampedConstPtr;

  using message_filters::Subscriber;

  typedef message_filters::TimeSynchronizer<ImageMsg, PoseStampedMsg> ExactSynchronizer2;
  typedef message_filters::sync_policies::ApproximateTime<ImageMsg, PoseStampedMsg> ApproxSyncPolicy2;
  typedef message_filters::Synchronizer<ApproxSyncPolicy2> ApproxTimeSynchronizer2;
  
  class GTFlowNodelet : public nodelet::Nodelet {
  public:
    void onInit() override;

  private:
    void callbackCameraInfo(CameraInfoConstPtr const &camInfoMsg);
    void callbackDepthAndPose(ImageConstPtr const &depthMsg,
                              PoseStampedConstPtr const &oposeMsg);
    // ---------- variables
    std::unique_ptr<image_transport::SubscriberFilter> imageSub_;
    std::unique_ptr<Subscriber<PoseStampedMsg> > poseSub_;
    //std::unique_ptr<Subscriber<ImageMsg> > imageSub_;
    std::unique_ptr<ExactSynchronizer2> sync_;
    std::unique_ptr<ApproxTimeSynchronizer2> approxSync_;
    ros::Subscriber cameraInfoSub_;
    CameraInfo      cameraInfo_;
    image_geometry::PinholeCameraModel        cameraModel_;
    std::vector<cv::Point3d> undistortedPoints_;
    bool            useApproxSync_{true};
    bool            hasInitPose_{false};
    Eigen::Isometry3d initPose_;
    ros::Time       lastTime_;
  };
}
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(gt_flow::GTFlowNodelet, nodelet::Nodelet)

#endif
