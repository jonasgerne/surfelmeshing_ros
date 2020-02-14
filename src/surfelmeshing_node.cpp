//
// Created by jonasgerstner on 08.02.20.
//
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "surfelmeshing_ros/surfelmeshing_server.h"
#include "libvis/libvis.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::TransformStamped> approx_policy;

int main(int argc, char **argv) {
    ros::init(argc, argv, "surfelmeshing");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    SurfelMeshingServer server(nh, nh_private);

    message_filters::Synchronizer<approx_policy> sync_(approx_policy(400));
    message_filters::Subscriber<sensor_msgs::Image> color_image_sub_, depth_image_sub_;
    message_filters::Subscriber<geometry_msgs::TransformStamped> transform_sub_;

    color_image_sub_.subscribe(nh, "image", 400);
    depth_image_sub_.subscribe(nh, "depth_image", 400);
    transform_sub_.subscribe(nh, "transform", 400);
    sync_.connectInput(color_image_sub_, depth_image_sub_, transform_sub_);
    sync_.registerCallback(boost::bind(&SurfelMeshingServer::messageCallback, &server, _1, _2, _3));

    ros::spin();
    return 0;
}

