//
// Created by jonasgerstner on 06.02.20.
//

#ifndef SURFELMESHING_ROS_CONVERSIONS_FROM_ROS_H
#define SURFELMESHING_ROS_CONVERSIONS_FROM_ROS_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/TransformStamped.h>
#include <opencv2/core/eigen.hpp>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <libvis/image.h>

namespace ROSConversions {
    inline vis::Image<vis::u16> convertDepthImage(const sensor_msgs::ImageConstPtr &image_input, double depth_scaling, bool apply_threshold, double max_depth) {
        cv_bridge::CvImageConstPtr cv_ptr;
        cv::Mat thresh;
        const cv::Mat* matptr;
        try {
            cv_ptr = cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (const cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
        if (apply_threshold){
            // apply a threshold for CARLA gt depth images as these go up to 1000.0m
            cv::threshold(cv_ptr->image, thresh, max_depth, 0.0, cv::THRESH_TOZERO_INV);
            matptr = &thresh;
        } else
            matptr = &cv_ptr->image;
        // NOTE: don't scale the images, cutoff is done later
        cv::Mat img_scaled_16u;
        cv::Mat(*matptr).convertTo(img_scaled_16u, CV_16UC1, depth_scaling);

        return vis::Image<vis::u16>(reinterpret_cast<vis::u32>(image_input->width),
                                    reinterpret_cast<vis::u32>(image_input->height),
                                    reinterpret_cast<vis::u16 *>(img_scaled_16u.data));
    }

    inline vis::Image<vis::Vec3u8> convertColorImage(const sensor_msgs::ImageConstPtr &image_input) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(image_input, sensor_msgs::image_encodings::RGB8);
        }
        catch (const cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }

        return vis::Image<vis::Vec3u8>(reinterpret_cast<vis::u32>(image_input->width),
                                       reinterpret_cast<vis::u32>(image_input->height),
                                       reinterpret_cast<vis::Vec3u8 *>(cv_ptr->image.data));
    }

    inline vis::SE3f convertTransformMsgToSE3f(const geometry_msgs::TransformStampedConstPtr &transform_msg) {
        Eigen::Quaterniond quatd;
        vis::Vector3d posd;

        tf2::fromMsg(transform_msg->transform.rotation, quatd);
        tf2::fromMsg(transform_msg->transform.translation, posd);

        // the template functions tf2::fromMsg are only instantiated for double types of Eigen vectors and quaternions
        // Reference: http://docs.ros.org/jade/api/tf2_eigen/html/tf2__eigen_8h.html
        Eigen::Quaternionf quatf = quatd.cast<float>();
        vis::Vector3f posf = posd.cast<float>();
        return vis::SE3f{quatf, posf};
    }

    inline vis::SE3f convertTfTransformToSE3f(const tf::Transform &transform_msg) {
        // Eigen uses this order: Quaternion (const Scalar &w, const Scalar &x, const Scalar &y, const Scalar &z)
        const tf::Quaternion &q = transform_msg.getRotation();
        const tf::Vector3 &v = transform_msg.getOrigin();
        Eigen::Quaterniond quat{q.getW(), q.getX(), q.getY(), q.getZ()};
        vis::Vector3d pos{v.getX(), v.getY(), v.getZ()};
        return vis::SE3f{quat.cast<float>(), pos.cast<float>()};
    }
}
#endif //SURFELMESHING_ROS_CONVERSIONS_FROM_ROS_H
