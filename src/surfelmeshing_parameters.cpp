//
// Created by jonasgerstner on 10.02.20.
//

#include "surfelmeshing_ros/surfelmeshing_parameters.h"
SurfelMeshingParameters::SurfelMeshingParameters(const ros::NodeHandle &nh_private){
    // camera
    nh_private.getParam("cam_fx", cam_fx);
    nh_private.getParam("cam_fy", cam_fy);
    nh_private.getParam("cam_cx", cam_cx);
    nh_private.getParam("cam_cy", cam_cy);
    nh_private.getParam("height", height);
    nh_private.getParam("width", width);
    nh_private.getParam("ego_to_cam", ego_to_cam);

    depth_scaling = 5000.0f;  // The default is for TUM RGB-D datasets.
    nh_private.param("depth_scaling", depth_scaling, depth_scaling); // "Input depth scaling: input_depth = depth_scaling * depth_in_meters. The default is for TUM RGB-D benchmark datasets."

    max_pose_interpolation_time_extent = 0.05f;
    nh_private.param("max_pose_interpolation_time_extent", max_pose_interpolation_time_extent, max_pose_interpolation_time_extent); // "The maximum time (in seconds) between the timestamp of a frame, and the preceding respectively succeeding trajectory pose timestamp, to interpolate the frame's pose. If this threshold is exceeded, the frame will be dropped since no close-enough pose information is available."

    start_frame = 0;
    nh_private.param("start_frame", start_frame, start_frame); // "First frame of the video to process."

    end_frame = std::numeric_limits<int>::max();
    nh_private.param("end_frame", end_frame, end_frame); // "If the video is longer, processing stops after end_frame."

    pyramid_level = 0;
    nh_private.param("pyramid_level", pyramid_level, pyramid_level); // "Specify the scale-space pyramid level to use. 0 uses the original sized images, 1 uses half the original resolution, etc."

    fps_restriction = 30;
    nh_private.param("restrict_fps_to", fps_restriction, fps_restriction); // "Restrict the frames per second to at most the given number."

    step_by_step_playback = false;
    nh_private.param("step_by_step_playback", step_by_step_playback, step_by_step_playback); // "Play back video frames step-by-step (do a step by pressing the Return key in the terminal)."

    invert_quaternions = false;
    nh_private.param("invert_quaternions", invert_quaternions, invert_quaternions); // "Invert the quaternions loaded from the poses file."

    // Surfel reconstruction parameters.
    max_surfel_count = 20 * 1000 * 1000;  // 20 million.
    nh_private.param("max_surfel_count", max_surfel_count, max_surfel_count); //"Maximum number of surfels. Determines the GPU memory requirements."

    sensor_noise_factor = 0.05f;
    nh_private.param("sensor_noise_factor", sensor_noise_factor, sensor_noise_factor); // "Sensor noise range extent as \"factor times the measured depth\". The real measurement is assumed to be in [(1 - sensor_noise_factor) * depth, (1 + sensor_noise_factor) * depth]."

    max_surfel_confidence = 5.0f;
    nh_private.param("max_surfel_confidence", max_surfel_confidence, max_surfel_confidence); // "Maximum value for the surfel confidence. Higher values enable more denoising, lower values faster adaptation to changes."

    regularizer_weight = 10.0f;
    nh_private.param("regularizer_weight", regularizer_weight, regularizer_weight); // "Weight for the regularization term (w_{reg} in the paper)."

    normal_compatibility_threshold_deg = 40.0f;
    nh_private.param("normal_compatibility_threshold_deg", normal_compatibility_threshold_deg, normal_compatibility_threshold_deg); // "Angle threshold (in degrees) for considering a measurement normal and a surfel normal to be compatible."

    regularization_frame_window_size = 30;
    nh_private.param("regularization_frame_window_size", regularization_frame_window_size, regularization_frame_window_size); // "Number of frames for which the regularization of a surfel is continued after it goes out of view."

    disable_blending = false;
    nh_private.param("disable_blending", disable_blending, disable_blending); // "Disable observation boundary blending."
    do_blending = !disable_blending;

    measurement_blending_radius = 12;
    nh_private.param("measurement_blending_radius", measurement_blending_radius, measurement_blending_radius); // "Radius for measurement blending in pixels."

    regularization_iterations_per_integration_iteration = 1;
    nh_private.param("regularization_iterations_per_integration_iteration",
                     regularization_iterations_per_integration_iteration, regularization_iterations_per_integration_iteration); // "Number of regularization (gradient descent) iterations performed per depth integration iteration. Set this to zero to disable regularization."

    radius_factor_for_regularization_neighbors = 2;
    nh_private.param("radius_factor_for_regularization_neighbors", radius_factor_for_regularization_neighbors, radius_factor_for_regularization_neighbors); // "Factor on the surfel radius for how far regularization neighbors can be away from a surfel."

    surfel_integration_active_window_size = std::numeric_limits<int>::max();
    nh_private.param("surfel_integration_active_window_size", surfel_integration_active_window_size, surfel_integration_active_window_size); // "Number of frames which need to pass before a surfel becomes inactive. If there are no loop closures, set this to a value larger than the dataset frame count to disable surfel deactivation."

    // Meshing parameters.
    max_angle_between_normals_deg = 90.0f;
    nh_private.param("max_angle_between_normals_deg", max_angle_between_normals_deg, max_angle_between_normals_deg); // "Maximum angle between normals of surfels that are connected by triangulation."
    max_angle_between_normals = M_PI / 180.0f * max_angle_between_normals_deg;

    min_triangle_angle_deg = 10.0f;
    nh_private.param("min_triangle_angle_deg", min_triangle_angle_deg, min_triangle_angle_deg); // "The meshing algorithm attempts to keep triangle angles larger than this."
    min_triangle_angle = M_PI / 180.0 * min_triangle_angle_deg;

    max_triangle_angle_deg = 170.0f;
    nh_private.param("max_triangle_angle_deg", max_triangle_angle_deg, max_triangle_angle_deg); // "The meshing algorithm attempts to keep triangle angles smaller than this."
    max_triangle_angle = M_PI / 180.0 * max_triangle_angle_deg;

    max_neighbor_search_range_increase_factor = 2.0f;
    nh_private.param("max_neighbor_search_range_increase_factor",
                     max_neighbor_search_range_increase_factor, max_neighbor_search_range_increase_factor); // "Maximum factor by which the surfel neighbor search range can be increased if the front neighbors are far away."

    long_edge_tolerance_factor = 1.5f;
    nh_private.param("long_edge_tolerance_factor", long_edge_tolerance_factor, long_edge_tolerance_factor); // "Tolerance factor over 'max_neighbor_search_range_increase_factor * surfel_radius' for deciding whether to remesh a triangle with long edges."

    synchronous_triangulation = false;
    nh_private.param("synchronous_meshing", synchronous_triangulation, synchronous_triangulation); // "Makes the meshing proceed synchronously to the surfel integration (instead of asynchronously)."
    asynchronous_triangulation = !synchronous_triangulation;

    full_meshing_every_frame = false;
    nh_private.param("full_meshing_every_frame", full_meshing_every_frame, full_meshing_every_frame); // "Instead of partial remeshing, performs full meshing in every frame. Only implemented for using together with --synchronous_meshing."

    full_retriangulation_at_end = false;
    nh_private.param("full_retriangulation_at_end", full_retriangulation_at_end, full_retriangulation_at_end); // "Performs a full retriangulation in the end (after the viewer closes, before the mesh is saved)."

    // Depth preprocessing parameters.
    max_depth = 3.0f;
    nh_private.param("max_depth", max_depth, max_depth); // "Maximum input depth in meters."

    depth_valid_region_radius = 333.0f;
    nh_private.param("depth_valid_region_radius", depth_valid_region_radius, depth_valid_region_radius); // "Radius of a circle (centered on the image center) with valid depth. Everything outside the circle is considered to be invalid. Used to discard biased depth at the corners of Kinect v1 depth images."

    observation_angle_threshold_deg = 85.0f;
    nh_private.param("observation_angle_threshold_deg", observation_angle_threshold_deg, observation_angle_threshold_deg); // "If the angle between the inverse observation direction and the measured surface normal is larger than this setting, the surface is discarded."

    depth_erosion_radius = 2;
    nh_private.param("depth_erosion_radius", depth_erosion_radius, depth_erosion_radius); // "Radius for depth map erosion (in [0, 3]). Useful to combat foreground fattening artifacts."

    median_filter_and_densify_iterations = 0;
    nh_private.param("median_filter_and_densify_iterations", median_filter_and_densify_iterations, median_filter_and_densify_iterations); // "Number of iterations of median filtering with hole filling. Disabled by default. Can be useful for noisy time-of-flight data."

    outlier_filtering_frame_count = 8;
    nh_private.param("outlier_filtering_frame_count", outlier_filtering_frame_count, outlier_filtering_frame_count); // "Number of other depth frames to use for outlier filtering of a depth frame. Supported values: 2, 4, 6, 8. Should be reduced if using low-frequency input."

    outlier_filtering_required_inliers = -1;
    nh_private.param("outlier_filtering_required_inliers", outlier_filtering_required_inliers, outlier_filtering_required_inliers); // "Number of required inliers for accepting a depth value in outlier filtering. With the default value of -1, all other frames (outlier_filtering_frame_count) must be inliers."

    bilateral_filter_sigma_xy = 3;
    nh_private.param("bilateral_filter_sigma_xy", bilateral_filter_sigma_xy, bilateral_filter_sigma_xy); // "sigma_xy for depth bilateral filtering, in pixels."

    bilateral_filter_radius_factor = 2.0f;
    nh_private.param("bilateral_filter_radius_factor", bilateral_filter_radius_factor, bilateral_filter_radius_factor); // "Factor on bilateral_filter_sigma_xy to define the kernel radius for depth bilateral filtering."

    bilateral_filter_sigma_depth_factor = 0.05;
    nh_private.param("bilateral_filter_sigma_depth_factor", bilateral_filter_sigma_depth_factor, bilateral_filter_sigma_depth_factor); // "Factor on the depth to compute sigma_depth for depth bilateral filtering."

    outlier_filtering_depth_tolerance_factor = 0.02f;
    nh_private.param("outlier_filtering_depth_tolerance_factor", outlier_filtering_depth_tolerance_factor, outlier_filtering_depth_tolerance_factor); // "Factor on the depth to define the size of the inlier region for outlier filtering."

    point_radius_extension_factor = 1.5f;
    nh_private.param("point_radius_extension_factor", point_radius_extension_factor, point_radius_extension_factor); // "Factor by which a point's radius is extended beyond the distance to its farthest neighbor."

    point_radius_clamp_factor = std::numeric_limits<float>::infinity();
    nh_private.param("point_radius_clamp_factor", point_radius_clamp_factor, point_radius_clamp_factor); // "Factor by which a point's radius can be larger than the distance to its closest neighbor (times sqrt(2)). Larger radii are clamped to this distance."

    // Octree parameters.
    max_surfels_per_node = 50;
    nh_private.param("max_surfels_per_node", max_surfels_per_node, max_surfels_per_node); // "Maximum number of surfels per octree node. Should only affect the runtime."

    // Render window parameters
    bool hide_new_surfel_splats = false;
    nh_private.param("hide_new_surfel_splats", hide_new_surfel_splats, hide_new_surfel_splats); // "Hides the splat rendering of new surfels which are not meshed yet."
    render_new_surfels_as_splats = !hide_new_surfel_splats;

    splat_half_extent_in_pixels = 3.0f;
    nh_private.param("splat_half_extent_in_pixels", splat_half_extent_in_pixels, splat_half_extent_in_pixels); // "Half splat quad extent in pixels."

    triangle_normal_shading = false;
    nh_private.param("triangle_normal_shading", triangle_normal_shading, triangle_normal_shading); // "Colors the mesh triangles based on their triangle normal.");

    bool hide_camera_frustum = false;
    nh_private.param("hide_camera_frustum", hide_camera_frustum, hide_camera_frustum); // "Hides the input camera frustum rendering.");
    render_camera_frustum = !hide_camera_frustum;

    bool exit_after_processing = false;
    nh_private.param("exit_after_processing", exit_after_processing, exit_after_processing); // "After processing the video, exit immediately instead of continuing to show the reconstruction.");
    show_result = !exit_after_processing;

    // File export parameters.
    export_mesh_path = "";
    nh_private.param<std::string>("export_mesh", export_mesh_path, export_mesh_path); // "Save the final mesh to the given path (as an OBJ file)."

    export_point_cloud_path = "";
    nh_private.param<std::string>("export_point_cloud", export_point_cloud_path, export_mesh_path); // "Save the final (surfel) point cloud to the given path (as a PLY file)."

    record_keyframes_path = "~/dummy_path";
    nh_private.param<std::string>("record_keyframes_path", record_keyframes_path, record_keyframes_path); // "Record keyframes for video recording to the given file. It is recommended to also set --step_by_step_playback and --show_result."

    playback_keyframes_path = "~/dummy_path";
    nh_private.param<std::string>("playback_keyframes_path", playback_keyframes_path, playback_keyframes_path); // "Play back keyframes for video recording from the given file."

    // Debug and evaluation parameters.
    create_video = false;
    nh_private.param("create_video", create_video, create_video); // "Records a video by writing screenshots frame-by-frame to the current working directory."

    debug_depth_preprocessing  = false;
    nh_private.param("debug_depth_preprocessing", debug_depth_preprocessing, debug_depth_preprocessing); // "Activates debug display of the depth maps at various stages of pre-processing."

    debug_neighbor_rendering = false;
    nh_private.param("debug_neighbor_rendering", debug_neighbor_rendering, debug_neighbor_rendering); // "Activates debug rendering of surfel regularization neighbors."

    debug_normal_rendering = false;
    nh_private.param("debug_normal_rendering", debug_normal_rendering, debug_normal_rendering); // "Activates debug rendering of surfel normal vectors."

    visualize_last_update_timestamp = false;
    nh_private.param("visualize_last_update_timestamp", visualize_last_update_timestamp, visualize_last_update_timestamp); // "Show a visualization of the surfel last update timestamps."

    visualize_creation_timestamp = false;
    nh_private.param("visualize_creation_timestamp", visualize_creation_timestamp, visualize_creation_timestamp); // "Show a visualization of the surfel creation timestamps."

    visualize_radii = false;
    nh_private.param("visualize_radii", visualize_radii, visualize_radii); // "Show a visualization of the surfel radii."

    visualize_surfel_normals = false;
    nh_private.param("visualize_surfel_normals", visualize_surfel_normals, visualize_surfel_normals); // "Show a visualization of the surfel normals."

    timings_log_path = "log";
    nh_private.param("log_timings", timings_log_path, timings_log_path); // "Log the timings to the given file."

}