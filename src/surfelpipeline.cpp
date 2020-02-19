//
// Created by jonasgerstner on 08.02.20.
//

#include "surfelmeshing_ros/surfelpipeline.h"

#include <memory>

SurfelPipeline::SurfelPipeline(SurfelMeshingParameters& param, const vis::PinholeCamera4f &depth_camera, const vis::Camera& scaled_camera,
                               vis::RGBDVideo<vis::Vec3u8, vis::u16>& rgbd_video) :
        param_(param),
        rgbd_video_(rgbd_video),
        depth_camera_(depth_camera),
        filtered_depth_buffer_A(param.height, param.width),
        filtered_depth_buffer_B(param.height, param.width),
        normals_buffer(param.height, param.width),
        radius_buffer(param.height, param.width),
        reconstruction(param.max_surfel_count, depth_camera_, vertex_buffer_resource,
                       neighbor_index_buffer_resource, normal_vertex_buffer_resource, nullptr),
        cuda_surfels_cpu_buffers(param.max_surfel_count),
        surfel_meshing(param.max_surfels_per_node, param.max_angle_between_normals, param.min_triangle_angle,
                       param.max_triangle_angle,
                       param.max_neighbor_search_range_increase_factor, param.long_edge_tolerance_factor,
                       param.regularization_frame_window_size, nullptr),
        latest_mesh_frame_index(0),
        latest_mesh_surfel_count(0),
        latest_mesh_triangle_count(0) {

    cudaEventCreate(&depth_image_upload_pre_event);
    cudaEventCreate(&depth_image_upload_post_event);
    cudaEventCreate(&color_image_upload_pre_event);
    cudaEventCreate(&color_image_upload_post_event);
    cudaEventCreate(&frame_start_event);
    cudaEventCreate(&bilateral_filtering_post_event);
    cudaEventCreate(&outlier_filtering_post_event);
    cudaEventCreate(&depth_erosion_post_event);
    cudaEventCreate(&normal_computation_post_event);
    cudaEventCreate(&preprocessing_end_event);
    cudaEventCreate(&frame_end_event);
    cudaEventCreate(&surfel_transfer_start_event);
    cudaEventCreate(&surfel_transfer_end_event);
    cudaEventCreate(&upload_finished_event);
    cudaStreamCreate(&stream);
    cudaStreamCreate(&upload_stream);

    // Allocate CUDA buffers.
    cudaHostAlloc(reinterpret_cast<void **>(&color_buffer_pagelocked),
                  param.height * param.width * sizeof(vis::Vec3u8),
                  cudaHostAllocWriteCombined);
    cudaHostAlloc(reinterpret_cast<void **>(&next_color_buffer_pagelocked),
                  param.height * param.width * sizeof(vis::Vec3u8),
                  cudaHostAllocWriteCombined);
    color_buffer.reset(new vis::CUDABuffer<vis::Vec3u8>(param.height, param.width));
    next_color_buffer.reset(new vis::CUDABuffer<vis::Vec3u8>(param.height, param.width));

    triangulation_thread = std::make_unique<vis::AsynchronousMeshing>(
            &surfel_meshing,
            &cuda_surfels_cpu_buffers,
            !param.timings_log_path.empty(),
            nullptr);
}

SurfelPipeline::~SurfelPipeline(){
    for (vis::u16* ptr : depth_buffers_pagelocked_cache) {
        cudaFreeHost(ptr);
    }
    for (std::pair<int, vis::u16*> item : frame_index_to_depth_buffer_pagelocked) {
        cudaFreeHost(item.second);
    }

    cudaFreeHost(color_buffer_pagelocked);
    cudaFreeHost(next_color_buffer_pagelocked);

    cudaEventDestroy(depth_image_upload_pre_event);
    cudaEventDestroy(depth_image_upload_post_event);
    cudaEventDestroy(color_image_upload_pre_event);
    cudaEventDestroy(color_image_upload_post_event);
    cudaEventDestroy(frame_start_event);
    cudaEventDestroy(bilateral_filtering_post_event);
    cudaEventDestroy(outlier_filtering_post_event);
    cudaEventDestroy(depth_erosion_post_event);
    cudaEventDestroy(normal_computation_post_event);
    cudaEventDestroy(preprocessing_end_event);
    cudaEventDestroy(frame_end_event);
    cudaEventDestroy(surfel_transfer_start_event);
    cudaEventDestroy(surfel_transfer_end_event);

    cudaEventDestroy(upload_finished_event);

    cudaStreamDestroy(stream);
    cudaStreamDestroy(upload_stream);
}


void SurfelPipeline::integrateImages(vis::usize frame_index) {
    vis::Timer frame_rate_timer("");  // "Frame rate timer (with I/O!)"

    // Since we do not want to measure the time for disk I/O, pre-load the
    // new images for this frame from disk here before starting the frame timer.
    if (frame_index + param_.outlier_filtering_frame_count / 2 + 1 < rgbd_video_.frame_count()) {
        rgbd_video_.depth_frame_mutable(frame_index + param_.outlier_filtering_frame_count / 2 + 1)->GetImage();
    }
    if (frame_index + 1 < rgbd_video_.frame_count()) {
        rgbd_video_.color_frame_mutable(frame_index + 1)->GetImage();
    }

    vis::ConditionalTimer complete_frame_timer("[Integration frame - measured on CPU]");

    cudaEventRecord(upload_finished_event, upload_stream);

    // Upload all frames up to (frame_index + outlier_filtering_frame_count / 2) to the GPU.
    for (vis::usize test_frame_index = frame_index;
         test_frame_index <=
         std::min(rgbd_video_.frame_count() - 1, frame_index + param_.outlier_filtering_frame_count / 2 + 1);
         ++test_frame_index) {
        if (frame_index_to_depth_buffer.count(test_frame_index)) {
            continue;
        }

        vis::u16 **pagelocked_ptr = &frame_index_to_depth_buffer_pagelocked[test_frame_index];
        vis::CUDABufferPtr<vis::u16> *buffer_ptr = &frame_index_to_depth_buffer[test_frame_index];

        if (depth_buffers_cache.empty()) {
            cudaHostAlloc(reinterpret_cast<void **>(pagelocked_ptr), param_.height * param_.width * sizeof(vis::u16),
                          cudaHostAllocWriteCombined);

            buffer_ptr->reset(new vis::CUDABuffer<vis::u16>(param_.height, param_.width));
        } else {
            *pagelocked_ptr = depth_buffers_pagelocked_cache.back();
            depth_buffers_pagelocked_cache.pop_back();

            *buffer_ptr = depth_buffers_cache.back();
            depth_buffers_cache.pop_back();
        }

        // Perform median filtering and densification.
        const vis::Image<vis::u16> *depth_map = rgbd_video_.depth_frame_mutable(test_frame_index)->GetImage().get();
        vis::Image<vis::u16> temp_depth_map;
        vis::Image<vis::u16> temp_depth_map_2;
        for (int iteration = 0; iteration < param_.median_filter_and_densify_iterations; ++iteration) {
            vis::Image<vis::u16> *target_depth_map = (depth_map == &temp_depth_map) ? &temp_depth_map_2 : &temp_depth_map;

            target_depth_map->SetSize(depth_map->size());
            MedianFilterAndDensifyDepthMap(*depth_map, target_depth_map);

            depth_map = target_depth_map;
        }
        if (param_.pyramid_level == 0) {
            memcpy(*pagelocked_ptr, depth_map->data(), param_.height * param_.width * sizeof(vis::u16));
        } else {
            if (param_.median_filter_and_densify_iterations > 0) {
                ROS_ERROR("Simultaneous downscaling and median filtering of depth maps is not implemented.");
                return;
            }

            vis::Image<vis::u16> downscaled_image(param_.width, param_.height);
            rgbd_video_.depth_frame_mutable(test_frame_index)->GetImage()->DownscaleUsingMedianWhileExcluding(0,
                                                                                                                     param_.width,
                                                                                                                     param_.height,
                                                                                                             &downscaled_image);
            memcpy(*pagelocked_ptr,
                   downscaled_image.data(),
                   param_.height * param_.width * sizeof(vis::u16));
        }
        cudaEventRecord(depth_image_upload_pre_event, upload_stream);
        (*buffer_ptr)->UploadAsync(upload_stream, *pagelocked_ptr);
        cudaEventRecord(depth_image_upload_post_event, upload_stream);
    }

    // Swap color image pointers and upload the next color frame to the GPU.
    std::swap(next_color_buffer, color_buffer);
    std::swap(next_color_buffer_pagelocked, color_buffer_pagelocked);
    if (param_.pyramid_level == 0) {
        memcpy(next_color_buffer_pagelocked,
               rgbd_video_.color_frame_mutable(frame_index + 1)->GetImage()->data(),
               param_.width * param_.height * sizeof(vis::Vec3u8));
    } else {
        memcpy(next_color_buffer_pagelocked,
               vis::ImagePyramid(rgbd_video_.color_frame_mutable(frame_index + 1).get(),
                            param_.pyramid_level).GetOrComputeResult()->data(),
               param_.width * param_.height * sizeof(vis::Vec3u8));
    }
    cudaEventRecord(color_image_upload_pre_event, upload_stream);
    next_color_buffer->UploadAsync(upload_stream, next_color_buffer_pagelocked);
    cudaEventRecord(color_image_upload_post_event, upload_stream);

    // If not enough neighboring frames are available for outlier filtering, go to the next frame.
    if (frame_index < static_cast<vis::usize>(param_.start_frame + param_.outlier_filtering_frame_count / 2) ||
        frame_index >= rgbd_video_.frame_count() - param_.outlier_filtering_frame_count / 2) {
        frame_rate_timer.Stop(false);
        complete_frame_timer.Stop(false);
        return;
    }

    // In the processing stream, wait for this frame's buffers to finish uploading in the upload stream.
    cudaStreamWaitEvent(stream, upload_finished_event, 0);

    // vis::ImageFramePtr<vis::Vec3u8, vis::SE3f> color_frame = rgbd_video_.color_frame_mutable(frame_index);
    vis::ImageFramePtr<vis::u16, vis::SE3f> input_depth_frame = rgbd_video_.depth_frame_mutable(frame_index);


    cudaEventRecord(frame_start_event, stream);

    vis::CUDABufferPtr<vis::u16> depth_buffer = frame_index_to_depth_buffer.at(frame_index);

    vis::Image<vis::u16> debug_depth(param_.width, param_.height);
    if (param_.debug_depth_preprocessing) {
        depth_buffer->DownloadAsync(stream, &debug_depth);
        io.Write("/home/jonasgerstner/Pictures/conversion/0_before_bilateral.png", debug_depth);
    }

    // Bilateral filtering and depth cutoff.
    BilateralFilteringAndDepthCutoffCUDA(
            stream,
            param_.bilateral_filter_sigma_xy,
            param_.bilateral_filter_sigma_depth_factor,
            0,
            param_.bilateral_filter_radius_factor,
            param_.depth_scaling * param_.max_depth,
            param_.depth_valid_region_radius,
            depth_buffer->ToCUDA(),
            &filtered_depth_buffer_A.ToCUDA());
    cudaEventRecord(bilateral_filtering_post_event, stream);

    if (param_.debug_depth_preprocessing){
        filtered_depth_buffer_A.DownloadAsync(stream, &debug_depth);
        cudaStreamSynchronize(stream);
        io.Write("/home/jonasgerstner/Pictures/conversion/2_bilateral.png", debug_depth);
    }

    // Depth outlier filtering.
    // Scale the poses to match the depth scaling. This is faster than scaling the depths of all pixels to match the poses.
    vis::SE3f input_depth_frame_scaled_frame_T_global = input_depth_frame->frame_T_global();
    input_depth_frame_scaled_frame_T_global.translation() =
            param_.depth_scaling * input_depth_frame_scaled_frame_T_global.translation();

    std::vector<const vis::CUDABuffer_<vis::u16> *> other_depths(param_.outlier_filtering_frame_count);
    std::vector<vis::SE3f> global_TR_others(param_.outlier_filtering_frame_count);
    std::vector<vis::CUDAMatrix3x4> others_TR_reference(param_.outlier_filtering_frame_count);

    for (int i = 0; i < param_.outlier_filtering_frame_count / 2; ++i) {
        int offset = i + 1;

        other_depths[i] = &frame_index_to_depth_buffer.at(frame_index - offset)->ToCUDA();
        global_TR_others[i] = rgbd_video_.depth_frame_mutable(frame_index - offset)->global_T_frame();
        global_TR_others[i].translation() = param_.depth_scaling * global_TR_others[i].translation();
        others_TR_reference[i] = vis::CUDAMatrix3x4(
                (input_depth_frame_scaled_frame_T_global * global_TR_others[i]).inverse().matrix3x4());

        int k = param_.outlier_filtering_frame_count / 2 + i;
        other_depths[k] = &frame_index_to_depth_buffer.at(frame_index + offset)->ToCUDA();
        global_TR_others[k] = rgbd_video_.depth_frame_mutable(frame_index + offset)->global_T_frame();
        global_TR_others[k].translation() = param_.depth_scaling * global_TR_others[k].translation();
        others_TR_reference[k] = vis::CUDAMatrix3x4(
                (input_depth_frame_scaled_frame_T_global * global_TR_others[k]).inverse().matrix3x4());
    }

    if (param_.outlier_filtering_required_inliers == -1 ||
            param_.outlier_filtering_required_inliers == param_.outlier_filtering_frame_count) {
        // Use a macro to pre-compile several versions of the template function.
#define CALL_OUTLIER_FUSION(other_frame_count) \
          vis::OutlierDepthMapFusionCUDA<other_frame_count + 1, vis::u16>( \
              stream, \
              param_.outlier_filtering_depth_tolerance_factor, \
              filtered_depth_buffer_A.ToCUDA(), \
              depth_camera_.parameters()[0], \
              depth_camera_.parameters()[1], \
              depth_camera_.parameters()[2], \
              depth_camera_.parameters()[3], \
              other_depths.data(), \
              others_TR_reference.data(), \
              &filtered_depth_buffer_B.ToCUDA())
        if (param_.outlier_filtering_frame_count == 2) {
            CALL_OUTLIER_FUSION(2);
        } else if (param_.outlier_filtering_frame_count == 4) {
            CALL_OUTLIER_FUSION(4);
        } else if (param_.outlier_filtering_frame_count == 6) {
            CALL_OUTLIER_FUSION(6);
        } else if (param_.outlier_filtering_frame_count == 8) {
            CALL_OUTLIER_FUSION(8);
        } else {
            ROS_FATAL_STREAM("Unsupported value for outlier_filtering_frame_count: " << param_.outlier_filtering_frame_count);
        }
#undef CALL_OUTLIER_FUSION
    } else {
        // Use a macro to pre-compile several versions of the template function.
#define CALL_OUTLIER_FUSION(other_frame_count) \
          vis::OutlierDepthMapFusionCUDA<other_frame_count + 1, vis::u16>( \
              stream, \
              param_.outlier_filtering_required_inliers, \
              param_.outlier_filtering_depth_tolerance_factor, \
              filtered_depth_buffer_A.ToCUDA(), \
              depth_camera_.parameters()[0], \
              depth_camera_.parameters()[1], \
              depth_camera_.parameters()[2], \
              depth_camera_.parameters()[3], \
              other_depths.data(), \
              others_TR_reference.data(), \
              &filtered_depth_buffer_B.ToCUDA())
        if (param_.outlier_filtering_frame_count == 2) {
            CALL_OUTLIER_FUSION(2);
        } else if (param_.outlier_filtering_frame_count == 4) {
            CALL_OUTLIER_FUSION(4);
        } else if (param_.outlier_filtering_frame_count == 6) {
            CALL_OUTLIER_FUSION(6);
        } else if (param_.outlier_filtering_frame_count == 8) {
            CALL_OUTLIER_FUSION(8);
        } else {
            ROS_FATAL_STREAM("Unsupported value for outlier_filtering_frame_count: " << param_.outlier_filtering_frame_count);
        }
#undef CALL_OUTLIER_FUSION
    }
    cudaEventRecord(outlier_filtering_post_event, stream);

    if (param_.debug_depth_preprocessing) {
        filtered_depth_buffer_B.DownloadAsync(stream, &debug_depth);
        cudaStreamSynchronize(stream);
        io.Write("/home/jonasgerstner/Pictures/conversion/4_filtered.png", debug_depth);
    }
    // Depth map erosion.
    if (param_.depth_erosion_radius > 0) {
        ErodeDepthMapCUDA(
                stream,
                param_.depth_erosion_radius,
                filtered_depth_buffer_B.ToCUDA(),
                &filtered_depth_buffer_A.ToCUDA());
    } else {
        CopyWithoutBorderCUDA(
                stream,
                filtered_depth_buffer_B.ToCUDA(),
                &filtered_depth_buffer_A.ToCUDA());
    }

    cudaEventRecord(depth_erosion_post_event, stream);

    // DEBUG: Show erosion result.
    if (param_.debug_depth_preprocessing) {
        filtered_depth_buffer_A.DownloadAsync(stream, &debug_depth);
        cudaStreamSynchronize(stream);
        io.Write("/home/jonasgerstner/Pictures/conversion/4_eroded.png", debug_depth);
    }

    ComputeNormalsAndDropBadPixelsCUDA(
            stream,
            param_.observation_angle_threshold_deg,
            param_.depth_scaling,
            depth_camera_.parameters()[0],
            depth_camera_.parameters()[1],
            depth_camera_.parameters()[2],
            depth_camera_.parameters()[3],
            filtered_depth_buffer_A.ToCUDA(),
            &filtered_depth_buffer_B.ToCUDA(),
            &normals_buffer.ToCUDA());

    cudaEventRecord(normal_computation_post_event, stream);

    // DEBUG: Show current depth map result.
    if (param_.debug_depth_preprocessing) {
        filtered_depth_buffer_B.DownloadAsync(stream, &debug_depth);
        cudaStreamSynchronize(stream);
        io.Write("/home/jonasgerstner/Pictures/conversion/5_dropped.png", debug_depth);
    }

    cudaEventRecord(preprocessing_end_event, stream);

    ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
            stream,
            param_.point_radius_extension_factor,
            param_.point_radius_clamp_factor,
            param_.depth_scaling,
            depth_camera_.parameters()[0],
            depth_camera_.parameters()[1],
            depth_camera_.parameters()[2],
            depth_camera_.parameters()[3],
            filtered_depth_buffer_B.ToCUDA(),
            &radius_buffer.ToCUDA(),
            &filtered_depth_buffer_A.ToCUDA());

    if (param_.debug_depth_preprocessing) {
        filtered_depth_buffer_A.DownloadAsync(stream, &debug_depth);
        cudaStreamSynchronize(stream);
        io.Write("/home/jonasgerstner/Pictures/conversion/7_isolate.png", debug_depth);
    }

    // Surfel reconstruction
    reconstruction.Integrate(
            stream,
            frame_index,
            param_.depth_scaling,
            &filtered_depth_buffer_A,
            normals_buffer,
            radius_buffer,
            *color_buffer,
            rgbd_video_.depth_frame_mutable(frame_index)->global_T_frame(),
            param_.sensor_noise_factor,
            param_.max_surfel_confidence,
            param_.regularizer_weight,
            param_.regularization_frame_window_size,
            param_.do_blending,
            param_.measurement_blending_radius,
            param_.regularization_iterations_per_integration_iteration,
            param_.radius_factor_for_regularization_neighbors,
            param_.normal_compatibility_threshold_deg,
            param_.surfel_integration_active_window_size);

    cudaEventRecord(frame_end_event, stream);

    ROS_DEBUG("Reconstruction done. Counting %d surfels.", reconstruction.surfel_count());

    /*
     * NOTE: SURFEL MESHING HANDLING
     * Transfer surfels to the CPU if no meshing is in progress,
     * if we expect that the next iteration will start very soon,
     * and for the last frame if the final result is needed.
    */
    cudaEventRecord(surfel_transfer_start_event, stream);
    triangulation_thread->LockInputData();
    cuda_surfels_cpu_buffers.LockWriteBuffers();

    reconstruction.TransferAllToCPU(
            stream,
            frame_index,
            &cuda_surfels_cpu_buffers);

    cudaEventRecord(surfel_transfer_end_event, stream);
    cudaStreamSynchronize(stream);

    // Notify the triangulation thread about new input data.
    // NOTE: It must be avoided to send this notification after the thread
    //       has already started working on the input (due to a previous
    //       notification), so do it while the write buffers are locked.
    //       Otherwise, the thread might later continue its
    //       next iteration before the write buffer was updated again,
    //       resulting in wrong data being used, in particular many surfels
    //       might be at (0, 0, 0).
    triangulation_thread->NotifyAboutNewInputSurfelsAlreadyLocked();

    cuda_surfels_cpu_buffers.UnlockWriteBuffers();
    triangulation_thread->UnlockInputData();

    cudaStreamSynchronize(stream);
    complete_frame_timer.Stop();

    // ### Profiling ###
    float elapsed_milliseconds;
    float frame_time_milliseconds = 0;
    float preprocessing_milliseconds = 0;
    float surfel_transfer_milliseconds = 0;

    // Synchronize with latest event
    cudaEventSynchronize(surfel_transfer_end_event);

    cudaEventSynchronize(depth_image_upload_post_event);
    cudaEventElapsedTime(&elapsed_milliseconds, depth_image_upload_pre_event, depth_image_upload_post_event);
    vis::Timing::addTime(vis::Timing::getHandle("Upload depth image"), 0.001 * elapsed_milliseconds);

    cudaEventSynchronize(color_image_upload_post_event);
    cudaEventElapsedTime(&elapsed_milliseconds, color_image_upload_pre_event, color_image_upload_post_event);
    vis::Timing::addTime(vis::Timing::getHandle("Upload color image"), 0.001 * elapsed_milliseconds);

    cudaEventElapsedTime(&elapsed_milliseconds, frame_start_event, bilateral_filtering_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    vis::Timing::addTime(vis::Timing::getHandle("Depth bilateral filtering"), 0.001 * elapsed_milliseconds);

    cudaEventElapsedTime(&elapsed_milliseconds, bilateral_filtering_post_event, outlier_filtering_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    vis::Timing::addTime(vis::Timing::getHandle("Depth outlier filtering"), 0.001 * elapsed_milliseconds);

    cudaEventElapsedTime(&elapsed_milliseconds, outlier_filtering_post_event, depth_erosion_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    vis::Timing::addTime(vis::Timing::getHandle("Depth erosion"), 0.001 * elapsed_milliseconds);

    cudaEventElapsedTime(&elapsed_milliseconds, depth_erosion_post_event, normal_computation_post_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    vis::Timing::addTime(vis::Timing::getHandle("Normal computation"), 0.001 * elapsed_milliseconds);

    cudaEventElapsedTime(&elapsed_milliseconds, normal_computation_post_event, preprocessing_end_event);
    frame_time_milliseconds += elapsed_milliseconds;
    preprocessing_milliseconds += elapsed_milliseconds;
    vis::Timing::addTime(vis::Timing::getHandle("Radius computation"), 0.001 * elapsed_milliseconds);

    cudaEventElapsedTime(&elapsed_milliseconds, preprocessing_end_event, frame_end_event);
    frame_time_milliseconds += elapsed_milliseconds;
    vis::Timing::addTime(vis::Timing::getHandle("Integration"), 0.001 * elapsed_milliseconds);

    vis::Timing::addTime(vis::Timing::getHandle("[CUDA frame]"), 0.001 * frame_time_milliseconds);


    cudaEventElapsedTime(&surfel_transfer_milliseconds, surfel_transfer_start_event, surfel_transfer_end_event);
    vis::Timing::addTime(vis::Timing::getHandle("Surfel transfer to CPU"), 0.001 * surfel_transfer_milliseconds);


    float data_association;
    float surfel_merging;
    float measurement_blending;
    float integration;
    float neighbor_update;
    float new_surfel_creation;
    float regularization;
    reconstruction.GetTimings(
            &data_association,
            &surfel_merging,
            &measurement_blending,
            &integration,
            &neighbor_update,
            &new_surfel_creation,
            &regularization);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - data_association"), 0.001 * data_association);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - surfel_merging"), 0.001 * surfel_merging);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - measurement_blending"), 0.001 * measurement_blending);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - integration"), 0.001 * integration);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - neighbor_update"), 0.001 * neighbor_update);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - new_surfel_creation"), 0.001 * new_surfel_creation);
    vis::Timing::addTime(vis::Timing::getHandle("Integration - regularization"), 0.001 * regularization);

    if (frame_index % kStatsLogInterval == 0) {
        ROS_INFO_STREAM(vis::Timing::print(vis::kSortByTotal));
    }

    if (!param_.timings_log_path.empty()) {
        timings_log << "frame " << frame_index << std::endl;
        timings_log << "-preprocessing " << preprocessing_milliseconds << std::endl;
        timings_log << "-data_association " << data_association << std::endl;
        timings_log << "-surfel_merging " << surfel_merging << std::endl;
        timings_log << "-measurement_blending " << measurement_blending << std::endl;
        timings_log << "-integration " << integration << std::endl;
        timings_log << "-neighbor_update " << neighbor_update << std::endl;
        timings_log << "-new_surfel_creation " << new_surfel_creation << std::endl;
        timings_log << "-regularization " << regularization << std::endl;
        timings_log << "-surfel_transfer " << surfel_transfer_milliseconds << std::endl;
        timings_log << "-surfel_count " << reconstruction.surfel_count() << std::endl;
    }

    // ### End-of-frame handling ###
    // Release frames which are no longer needed.
    int last_frame_in_window = frame_index - param_.outlier_filtering_frame_count / 2;
    if (last_frame_in_window >= 0) {
        rgbd_video_.color_frame_mutable(last_frame_in_window)->ClearImageAndDerivedData();
        rgbd_video_.depth_frame_mutable(last_frame_in_window)->ClearImageAndDerivedData();
        depth_buffers_pagelocked_cache.push_back(frame_index_to_depth_buffer_pagelocked.at(last_frame_in_window));
        frame_index_to_depth_buffer_pagelocked.erase(last_frame_in_window);
        depth_buffers_cache.push_back(frame_index_to_depth_buffer.at(last_frame_in_window));
        frame_index_to_depth_buffer.erase(last_frame_in_window);
    }
}

void SurfelPipeline::MedianFilterAndDensifyDepthMap(const vis::Image<vis::u16> &input, vis::Image<vis::u16> *output) {
    std::vector<vis::u16> values;

    constexpr int kRadius = 1;
    constexpr int kMinNeighbors = 2;

    for (int y = 0; y < static_cast<int>(input.height()); ++y) {
        for (int x = 0; x < static_cast<int>(input.width()); ++x) {
            values.clear();

            int dy_end = std::min<int>(input.height() - 1, y + kRadius);
            for (int dy = std::max<int>(0, static_cast<int>(y) - kRadius);
                 dy <= dy_end;
                 ++dy) {
                int dx_end = std::min<int>(input.width() - 1, x + kRadius);
                for (int dx = std::max<int>(0, static_cast<int>(x) - kRadius);
                     dx <= dx_end;
                     ++dx) {
                    if (input(dx, dy) != 0) {
                        values.push_back(input(dx, dy));
                    }
                }
            }

            if (values.size() >= kMinNeighbors) {
                std::sort(values.begin(), values.end());  // NOTE: slow, need to get center element only
                if (values.size() % 2 == 0) {
                    // Take the element which is closer to the average.
                    float sum = 0;
                    for (vis::u16 value : values) {
                        sum += value;
                    }
                    float average = sum / values.size();

                    float prev_diff = std::fabs(values[values.size() / 2 - 1] - average);
                    float next_diff = std::fabs(values[values.size() / 2] - average);
                    (*output)(x, y) = (prev_diff < next_diff) ? values[values.size() / 2 - 1] : values[values.size() /
                                                                                                       2];
                } else {
                    (*output)(x, y) = values[values.size() / 2];
                }
            } else {
                (*output)(x, y) = input(x, y);
            }
        }
    }
}

bool SurfelPipeline::prepareOutput(vis::usize frame_index) {
    cudaEventRecord(surfel_transfer_start_event, stream);
    triangulation_thread->LockInputData();

    cuda_surfels_cpu_buffers.LockWriteBuffers();

    reconstruction.TransferAllToCPU(
            stream,
            frame_index,
            &cuda_surfels_cpu_buffers);

    cudaEventRecord(surfel_transfer_end_event, stream);
    cudaStreamSynchronize(stream);

    // Notify the triangulation thread about new input data.
    // NOTE: It must be avoided to send this notification after the thread
    //       has already started working on the input (due to a previous
    //       notification), so do it while the write buffers are locked.
    //       Otherwise, the thread might later continue its
    //       next iteration before the write buffer was updated again,
    //       resulting in wrong data being used, in particular many surfels
    //       might be at (0, 0, 0).
    triangulation_thread->NotifyAboutNewInputSurfelsAlreadyLocked();

    cuda_surfels_cpu_buffers.UnlockWriteBuffers();
    triangulation_thread->UnlockInputData();

    cudaStreamSynchronize(stream);
}

std::shared_ptr<vis::Mesh3fCu8> SurfelPipeline::getMesh() {
    LOG(INFO) << "Waiting for final mesh ...";
    while (!triangulation_thread->all_work_done())
        usleep(0);

    CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size());
    std::shared_ptr<vis::Mesh3fCu8> mesh(new vis::Mesh3fCu8());
    // Also use the positions from the surfel_meshing such that positions
    // and the mesh are from a consistent state.
    surfel_meshing.ConvertToMesh3fCu8(mesh.get());

    vis::CUDABuffer<float> position_buffer(1, 3 * reconstruction.surfels_size());
    vis::CUDABuffer<vis::u8> color_buffer(1, 3 * reconstruction.surfels_size());
    reconstruction.ExportVertices(stream, &position_buffer, &color_buffer);
    float* position_buffer_cpu = new float[3 * reconstruction.surfels_size()];
    vis::u8* color_buffer_cpu = new vis::u8[3 * reconstruction.surfels_size()];
    position_buffer.DownloadAsync(stream, position_buffer_cpu);
    color_buffer.DownloadAsync(stream, color_buffer_cpu);
    cudaStreamSynchronize(stream);
    vis::usize index = 0;
    int count_nans = 0;
    CHECK_EQ(mesh->vertices()->size(), reconstruction.surfel_count());

    for (vis::usize i = 0; i < reconstruction.surfels_size(); ++i) {
        if (isnan(position_buffer_cpu[3 * i + 0])) {
            ++count_nans;
            continue;
        }

        vis::Point3fC3u8* point = &(*mesh->vertices_mutable())->at(index);
        point->color() = vis::Vec3u8(color_buffer_cpu[3 * i + 0],
                                color_buffer_cpu[3 * i + 1],
                                color_buffer_cpu[3 * i + 2]);
        ++index;
    }

    CHECK_EQ(index, mesh->vertices()->size());

    delete[] color_buffer_cpu;
    delete[] position_buffer_cpu;

    return mesh;
}

// Saves the reconstructed surfels as a point cloud in PLY format.
bool SurfelPipeline::SavePointCloudAsPLY(){
    // CHECK_EQ(surfel_meshing.surfels().size(), reconstruction.surfels_size());

    vis::Point3fCloud cloud;
    surfel_meshing.ConvertToPoint3fCloud(&cloud);

    vis::Point3fC3u8NfCloud final_cloud(cloud.size());
    vis::usize index = 0;
    for (vis::usize i = 0; i < cloud.size(); ++i) {
        final_cloud[i].position() = cloud[i].position();
        final_cloud[i].color() = vis::Vec3u8(255, 255, 255);  // TODO: Export colors as well.
        while (surfel_meshing.surfels()[index].node() == nullptr) {
            ++index;
        }
        final_cloud[i].normal() = surfel_meshing.surfels()[index].normal();
        ++index;
    }
    final_cloud.WriteAsPLY(param_.export_point_cloud_path);
            LOG(INFO) << "Wrote " << param_.export_point_cloud_path << ".";
    return true;
}
