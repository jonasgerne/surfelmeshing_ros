//
// Created by jonasgerstner on 06.02.20.
//

#include "surfelmeshing_ros/surfelmeshing_server.h"

SurfelMeshingServer::SurfelMeshingServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh), 
      nh_private_(nh_private),
      param_(nh_private),
      current_frame_(0),
      save_once(false)
      {

    rosmesh_pub_ = nh_private_.advertise<mesh_msgs::TriangleMeshStamped>("rosmesh", 1, true);

    float camera_parameters[4] = {param_.cam_fx, param_.cam_fy, param_.cam_cx + 0.5f, param_.cam_cy + 0.5f};
    rgbd_video.color_camera_mutable()->reset(new vis::PinholeCamera4f(param_.width, param_.height, camera_parameters));
    rgbd_video.depth_camera_mutable()->reset(new vis::PinholeCamera4f(param_.width, param_.height, camera_parameters));

    // Get potentially scaled depth camera as pinhole camera, determine input size.
    vis::Camera &generic_depth_camera = *rgbd_video.depth_camera();
    scaled_camera.reset(generic_depth_camera.Scaled(1.0f / powf(2, param_.pyramid_level)));

    auto &depth_camera = reinterpret_cast<const vis::PinholeCamera4f &>(*scaled_camera);

//    render_window.reset(new vis::SurfelMeshingRenderWindow(param_.render_new_surfels_as_splats,
//                                                           param_.splat_half_extent_in_pixels,
//                                                           param_.triangle_normal_shading,
//                                                           param_.render_camera_frustum));

    pipeline_ptr.reset(new SurfelPipeline(param_, depth_camera, *scaled_camera, rgbd_video));

    generate_mesh_srv_ = nh_private_.advertiseService("generate_mesh", &SurfelMeshingServer::generateMeshCallback,this);

    save_ply_srv_ = nh_private_.advertiseService("save_ply", &SurfelMeshingServer::savePLYCallback, this);

    setImuCam(param_.ego_to_cam);
    ROS_INFO("Setup Server.");
}

void SurfelMeshingServer::messageCallback(const sensor_msgs::ImageConstPtr& color_image,
                                          const sensor_msgs::ImageConstPtr& depth_image,
                                          const geometry_msgs::TransformStampedConstPtr& transform_msg) {
    ROS_INFO("messageCallback");
    tf::StampedTransform imu_transform;
    tf::transformStampedMsgToTF(*transform_msg, imu_transform);

    tf::Transform world_cam_tf;
    world_cam_tf = imu_transform * imu_cam_;

    // auto global_T_frame = ROSConversions::convertTransformMsgToSE3f(transform_msg);
    auto global_T_frame = ROSConversions::convertTfTransformToSE3f(world_cam_tf);

    auto image = std::make_shared<vis::Image<vis::Vec3u8>>(ROSConversions::convertColorImage(color_image));
    vis::ImageFramePtr<vis::Vec3u8, vis::SE3f> image_frame(new vis::ImageFrame<vis::Vec3u8, vis::SE3f>(image));
    image_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video.color_frames_mutable()->push_back(image_frame);

    auto depth = std::make_shared<vis::Image<vis::u16>>(ROSConversions::convertDepthImage(depth_image, param_.depth_scaling, param_.apply_threshold, param_.depth_threshold));
    vis::ImageFramePtr<vis::u16, vis::SE3f> depth_frame(new vis::ImageFrame<vis::u16, vis::SE3f>(depth));
    depth_frame->SetGlobalTFrame(global_T_frame);
    rgbd_video.depth_frames_mutable()->push_back(depth_frame);
    if(!save_once){
        io.Write("/home/jonasgerstner/Pictures/conversion/first.png", *depth);
        save_once = true;
    }


    if(rgbd_video.frame_count() > param_.outlier_filtering_frame_count / 2 + 1) {
        pipeline_ptr->integrateImages(current_frame_);
        ++current_frame_;
    }
}

bool SurfelMeshingServer::generateMeshCallback(std_srvs::Empty::Request& /*request*/,
                                               std_srvs::Empty::Response& /*response*/) {  // NOLINT
    return generateMeshToolsMesh();
}

bool SurfelMeshingServer::savePLYCallback(std_srvs::Empty::Request& /*request*/,
                                               std_srvs::Empty::Response& /*response*/) {  // NOLINT
    return pipeline_ptr->SavePointCloudAsPLY();
}

bool SurfelMeshingServer::generateMeshToolsMesh() {
    pipeline_ptr->prepareOutput(current_frame_-1);
    auto mesh = pipeline_ptr->getMesh();

    mesh_msgs::TriangleMeshStamped mesh_msg_stmp;

    ROSConversions::generateMeshToolsMessage(mesh, &mesh_msg_stmp);
    mesh_msg_stmp.header.frame_id = "world";

    rosmesh_pub_.publish(mesh_msg_stmp);
    return true;
}

bool SurfelMeshingServer::setImuCam(const std::string &transform_str){
    // TODO: add error handling, probably use strtof instead of scanf
    float x, y, z, qx, qy, qz, qw;
    sscanf(transform_str.c_str(), "%f %f %f %f %f %f %f", &x, &y, &z, &qx, &qy, &qz, &qw);
    tf::Vector3 vec(x, y, z);
    tf::Quaternion quat(qx, qy, qz, qw);
    imu_cam_.setOrigin(vec);
    imu_cam_.setRotation(quat);
    return true;
}



