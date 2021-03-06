//
// Created by jonasgerstner on 06.02.20.
//

#ifndef SURFELMESHING_ROS_SURFELMESHING_SERVER_H
#define SURFELMESHING_ROS_SURFELMESHING_SERVER_H

#include <ros/ros.h>
#include <string>
#include <limits>
#include <mesh_msgs/TriangleMeshStamped.h>
#include <std_srvs/Empty.h>
#include <tf/transform_datatypes.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>
#include <pcl/PolygonMesh.h>

#include <libvis/libvis.h>
#include <libvis/point_cloud.h>
#include <libvis/rgbd_video.h>
#include <libvis/sophus.h>
#include <libvis/image.h>
#include <libvis/image_frame.h>
#include <libvis/eigen.h>

#include "surfelmeshing_ros/conversions_from_ros.h"
#include "surfelmeshing_ros/conversion_to_ros.h"
#include "surfelmeshing_ros/surfelmeshing_parameters.h"
#include "surfelmeshing_ros/surfelpipeline.h"

class SurfelMeshingServer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SurfelMeshingServer(const ros::NodeHandle &nh,
                        const ros::NodeHandle &nh_private);

    void messageCallback(const sensor_msgs::ImageConstPtr &, const sensor_msgs::ImageConstPtr &,
                         const geometry_msgs::TransformStampedConstPtr &);
    bool generateMeshCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
    bool generatePCLMeshCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
    bool savePLYCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&);


protected:
    // ROS Nodehandle
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    SurfelMeshingParameters param_;

    //Publisher
    ros::Publisher rosmesh_pub_;
    ros::Publisher pcl_mesh_pub_;

    // Services
    ros::ServiceServer generate_mesh_srv_;
    ros::ServiceServer generate_pcl_mesh_srv_;
    ros::ServiceServer save_ply_srv_;

    // SurfelMeshing
    vis::RGBDVideo<vis::Vec3u8, vis::u16> rgbd_video;

    std::shared_ptr<vis::SurfelMeshingRenderWindow> render_window;

    std::unique_ptr<vis::Camera> scaled_camera;

    std::shared_ptr<SurfelPipeline> pipeline_ptr;

    int frame_count;
    vis::usize current_frame_;

    tf::StampedTransform imu_cam_;

    bool generateMeshToolsMesh();
    bool generatePCLPolygonMesh();

    bool setImuCam(const std::string &transform_str);

    vis::ImageIOLibPng io;
    bool save_once;


};
#endif //SURFELMESHING_ROS_SURFELMESHING_SERVER_H
