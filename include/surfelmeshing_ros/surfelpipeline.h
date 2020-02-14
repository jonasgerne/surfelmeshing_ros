//
// Created by jonasgerstner on 08.02.20.
//

#ifndef SURFELMESHING_ROS_SURFELMESHING_H
#define SURFELMESHING_ROS_SURFELMESHING_H

#include <boost/filesystem.hpp>
#include <cuda_runtime.h>
#include <libvis/libvis.h>
#include <libvis/image_display.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/mesh_opengl.h>
#include <libvis/opengl_context.h>
#include <libvis/point_cloud.h>
#include <libvis/image_display.h>
#include <libvis/rgbd_video.h>
#include <libvis/sophus.h>
#include <libvis/timing.h>
#include <libvis/image.h>
#include <libvis/image_frame.h>
#include <libvis/eigen.h>
#include <signal.h>
#include <spline_library/splines/uniform_cr_spline.h>
#include <chrono>
#include <thread>

#include "surfel_meshing/asynchronous_meshing.h"
#include "surfel_meshing/cuda_depth_processing.cuh"
#include "surfel_meshing/cuda_surfel_reconstruction.h"
//#include "surfel_meshing/surfel_meshing_render_window.h"
#include "surfel_meshing/surfel.h"
#include "surfel_meshing/surfel_meshing.h"
#include "surfelmeshing_ros/surfelmeshing_parameters.h"
class SurfelPipeline{
public:
    SurfelPipeline(SurfelMeshingParameters&, const vis::PinholeCamera4f&, const vis::Camera&, vis::RGBDVideo<vis::Vec3u8, vis::u16>&);
    void integrateImages(vis::usize frame_index);
    static void MedianFilterAndDensifyDepthMap(const vis::Image<vis::u16>&, vis::Image<vis::u16>*);
    std::shared_ptr<vis::Mesh3fCu8> getMesh();

    vis::OpenGLContext opengl_context;
    cudaGraphicsResource_t vertex_buffer_resource = nullptr;
    cudaGraphicsResource_t neighbor_index_buffer_resource = nullptr;
    cudaGraphicsResource_t normal_vertex_buffer_resource = nullptr;

    // Initialize CUDA streams.
    cudaStream_t stream;
    cudaStream_t upload_stream;

    // Initialize CUDA events.
    cudaEvent_t depth_image_upload_pre_event;
    cudaEvent_t depth_image_upload_post_event;
    cudaEvent_t color_image_upload_pre_event;
    cudaEvent_t color_image_upload_post_event;
    cudaEvent_t frame_start_event;
    cudaEvent_t bilateral_filtering_post_event;
    cudaEvent_t outlier_filtering_post_event;
    cudaEvent_t depth_erosion_post_event;
    cudaEvent_t normal_computation_post_event;
    cudaEvent_t preprocessing_end_event;
    cudaEvent_t frame_end_event;
    cudaEvent_t surfel_transfer_start_event;
    cudaEvent_t surfel_transfer_end_event;

    cudaEvent_t upload_finished_event;

protected:
    SurfelMeshingParameters& param_;
    vis::RGBDVideo<vis::Vec3u8, vis::u16>& rgbd_video_;
    const vis::PinholeCamera4f& depth_camera_;
    std::shared_ptr<vis::SurfelMeshingRenderWindow> render_window_;

    // CUDA
    std::unordered_map<int, vis::u16 *> frame_index_to_depth_buffer_pagelocked;
    std::unordered_map<int, vis::CUDABufferPtr<vis::u16>> frame_index_to_depth_buffer;

    vis::CUDABuffer<vis::u16> filtered_depth_buffer_A;
    vis::CUDABuffer<vis::u16> filtered_depth_buffer_B;

    vis::CUDABuffer<float2> normals_buffer;
    vis::CUDABuffer<float> radius_buffer;

    vis::Vec3u8 *color_buffer_pagelocked;
    vis::Vec3u8 *next_color_buffer_pagelocked;

    std::shared_ptr<vis::CUDABuffer<vis::Vec3u8>> color_buffer;
    std::shared_ptr<vis::CUDABuffer<vis::Vec3u8>> next_color_buffer;

    std::vector<vis::u16 *> depth_buffers_pagelocked_cache;
    std::vector<vis::CUDABufferPtr<vis::u16>> depth_buffers_cache;

    vis::CUDASurfelReconstruction reconstruction;
    vis::CUDASurfelsCPU cuda_surfels_cpu_buffers;
    vis::SurfelMeshing surfel_meshing;

    std::unique_ptr<vis::AsynchronousMeshing> triangulation_thread;

    vis::u32 latest_mesh_frame_index;
    vis::u32 latest_mesh_surfel_count;
    vis::usize latest_mesh_triangle_count;
    bool triangulation_in_progress;

    std::ostringstream timings_log;
    std::ostringstream meshing_timings_log;

    static constexpr int kStatsLogInterval = 200;
};
#endif //SURFELMESHING_ROS_SURFELMESHING_H
