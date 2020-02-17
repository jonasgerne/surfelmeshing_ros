//
// Created by jonasgerstner on 10.02.20.
//

#ifndef SURFELMESHING_ROS_SURFELMESHING_PARAMETERS_H
#define SURFELMESHING_ROS_SURFELMESHING_PARAMETERS_H

#include <ros/ros.h>

class SurfelMeshingParameters{
public:
    SurfelMeshingParameters(const ros::NodeHandle &nh_private);

    // Camera
    float cam_fx;
    float cam_fy;
    float cam_cx;
    float cam_cy;
    int height;
    int width;

    std::string ego_to_cam;

    // Dataset playback parameters
    float depth_scaling;  // The default is for TUM RGB-D datasets.
    bool apply_threshold;
    float depth_threshold;
    float max_pose_interpolation_time_extent;
    int start_frame;
    int end_frame;
    int pyramid_level;
    int fps_restriction;
    bool step_by_step_playback;
    bool invert_quaternions;


    // Surfel reconstruction parameters.
    int max_surfel_count;  // 20 million.
    float sensor_noise_factor;
    float max_surfel_confidence;
    float regularizer_weight;
    float normal_compatibility_threshold_deg;
    int regularization_frame_window_size;
    bool disable_blending;
    bool do_blending;
    int measurement_blending_radius;
    int regularization_iterations_per_integration_iteration;
    float radius_factor_for_regularization_neighbors;
    int surfel_integration_active_window_size;

    // Meshing parameters.
    float max_angle_between_normals_deg;
    float max_angle_between_normals;
    float min_triangle_angle_deg;
    float min_triangle_angle;
    float max_triangle_angle_deg;
    float max_triangle_angle;
    float max_neighbor_search_range_increase_factor;
    float long_edge_tolerance_factor;
    bool synchronous_triangulation;
    bool asynchronous_triangulation;
    bool full_meshing_every_frame;
    bool full_retriangulation_at_end;

    // Depth preprocessing parameters.
    float max_depth;
    float depth_valid_region_radius;
    float observation_angle_threshold_deg;
    int depth_erosion_radius;
    int median_filter_and_densify_iterations;
    int outlier_filtering_frame_count;
    int outlier_filtering_required_inliers;
    float bilateral_filter_sigma_xy;
    float bilateral_filter_radius_factor;
    float bilateral_filter_sigma_depth_factor;
    float outlier_filtering_depth_tolerance_factor;
    float point_radius_extension_factor;
    float point_radius_clamp_factor;

    // Octree parameters.
    int max_surfels_per_node;

    // Render window parameters
    bool render_new_surfels_as_splats;
    float splat_half_extent_in_pixels;
    bool triangle_normal_shading;
    bool render_camera_frustum;
    bool show_result;

    // File export parameters.
    std::string export_mesh_path;
    std::string export_point_cloud_path;
    std::string record_keyframes_path;
    std::string playback_keyframes_path;

    // Debug and evaluation parameters.
    bool create_video;
    bool debug_depth_preprocessing;
    bool debug_neighbor_rendering;
    bool debug_normal_rendering;
    bool visualize_last_update_timestamp;
    bool visualize_creation_timestamp;
    bool visualize_radii;
    bool visualize_surfel_normals;
    std::string timings_log_path;

    // Required input paths.
    std::string dataset_folder_path;
    std::string trajectory_filename;
};
#endif //SURFELMESHING_ROS_SURFELMESHING_PARAMETERS_H
