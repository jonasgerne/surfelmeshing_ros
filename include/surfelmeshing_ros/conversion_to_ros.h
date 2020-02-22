//
// Created by jonasgerstner on 11.02.20.
//

#ifndef SURFELMESHING_ROS_CONVERSION_TO_ROS_H
#define SURFELMESHING_ROS_CONVERSION_TO_ROS_H

#include <type_traits>

#include "libvis/libvis.h"
#include "libvis/mesh.h"
#include "libvis/point_cloud.h"
#include <mesh_msgs/TriangleMeshStamped.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

namespace ROSConversions{
    inline void fromLibvis(const vis::PointCloud<vis::Point3f>& cloud, mesh_msgs::TriangleMesh& mesh_msg) { //3fCloud& cloud) {
        for (vis::usize i = 0; i < cloud.size(); ++ i) {
            const vis::Point3f& point = cloud.data()[i];
            geometry_msgs::Point point_msg;
            point_msg.x = point.position().x();
            point_msg.y = point.position().y();
            point_msg.z = point.position().z();
            mesh_msg.vertices.push_back(point_msg);
        }
    }

    inline void fromLibvis(const vis::PointCloud<vis::Point3fCu8>& cloud, mesh_msgs::TriangleMesh& mesh_msg) {
        mesh_msg.vertex_colors.reserve(cloud.size());

        constexpr float normalization_factor = 1.0f / std::numeric_limits<vis::u8>::max();

        for (vis::usize i = 0; i < cloud.size(); ++ i) {
            const vis::Point3fCu8& point = cloud.data()[i];
            geometry_msgs::Point point_msg;
            point_msg.x = point.position().x();
            point_msg.y = point.position().y();
            point_msg.z = point.position().z();
            mesh_msg.vertices.push_back(point_msg);

            // Note: u8 is a greyscale image
            std_msgs::ColorRGBA color_msg;
            color_msg.r = point.color() * normalization_factor;
            color_msg.g = point.color() * normalization_factor;
            color_msg.b = point.color() * normalization_factor;
            color_msg.a = 1.0f;
            mesh_msg.vertex_colors.push_back(color_msg);
        }
    }

    inline void fromLibvis(const vis::PointCloud<vis::Point3fC3u8>& cloud, mesh_msgs::TriangleMesh& mesh_msg) {
        mesh_msg.vertex_colors.reserve(cloud.size());

        constexpr float normalization_factor = 1.0f / std::numeric_limits<vis::u8>::max();

        for (vis::usize i = 0; i < cloud.size(); ++ i) {
            const vis::Point3fC3u8& point = cloud.data()[i];

            geometry_msgs::Point point_msg;
            point_msg.x = point.position().x();
            point_msg.y = point.position().y();
            point_msg.z = point.position().z();
            mesh_msg.vertices.push_back(point_msg);

            std_msgs::ColorRGBA color_msg;
            color_msg.r = point.color().x() * normalization_factor;
            color_msg.g = point.color().y() * normalization_factor;
            color_msg.b = point.color().z() * normalization_factor;
            color_msg.a = 1.0f;
            mesh_msg.vertex_colors.push_back(color_msg);
        }
    }

    inline void fromLibvis(const vis::PointCloud<vis::Point3fC3u8Nf>& cloud, mesh_msgs::TriangleMesh& mesh_msg) {
        mesh_msg.vertex_normals.reserve(cloud.size());
        mesh_msg.vertex_colors.reserve(cloud.size());

        constexpr float normalization_factor = 1.0f / std::numeric_limits<vis::u8>::max();

        for (vis::usize i = 0; i < cloud.size(); ++ i) {
            const vis::Point3fC3u8Nf& point = cloud.data()[i];

            geometry_msgs::Point point_msg;
            point_msg.x = point.position().x();
            point_msg.y = point.position().y();
            point_msg.z = point.position().z();
            mesh_msg.vertices.push_back(point_msg);

            std_msgs::ColorRGBA color_msg;
            color_msg.r = point.color().x() * normalization_factor;
            color_msg.g = point.color().x() * normalization_factor;
            color_msg.b = point.color().x() * normalization_factor;
            color_msg.a = 1.0f;
            mesh_msg.vertex_colors.push_back(color_msg);

            geometry_msgs::Point normal_msg;
            normal_msg.x = point.position().x();
            normal_msg.y = point.position().y();
            normal_msg.z = point.position().z();
            mesh_msg.vertex_normals.push_back(normal_msg);
        }
    }

    inline void generateMeshToolsMessage(const std::shared_ptr<vis::Mesh3fCu8>& mesh, mesh_msgs::TriangleMeshStamped* mesh_msg_stmpd){
        mesh_msg_stmpd->header.stamp = ros::Time::now();

        mesh_msgs::TriangleMesh mesh_msg;

        size_t num_points = mesh->vertices()->size();

        mesh_msg.vertices.reserve(num_points);

        fromLibvis((*mesh->vertices()), mesh_msg);
        mesh_msg.triangles.reserve(num_points/3);

        for (const vis::Triangle<vis::u32>& triangle : mesh->triangles()){
            mesh_msgs::TriangleIndices indices_msg;
            for (int j = 0; j < 3; j++)
                indices_msg.vertex_indices[j] = static_cast<uint32_t>(triangle.index(j));
            mesh_msg.triangles.push_back(indices_msg);
        }

        mesh_msg_stmpd->mesh = mesh_msg;
    }

    inline void fromLibvis(const std::shared_ptr<vis::Mesh3fCu8>& mesh, pcl::PolygonMesh* polygon_mesh_ptr) {
        // Constructing the vertices pointcloud
        pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
        std::vector<pcl::Vertices> polygons;

        // add points
        pointcloud.reserve(mesh->vertices()->size());
        for (vis::usize i = 0; i < mesh->vertices()->size(); ++ i) {
            const vis::Point3fC3u8& point = mesh->vertices()->data()[i];
            pcl::PointXYZRGB pt;
            pt.x = static_cast<float>(point.position().x());
            pt.y = static_cast<float>(point.position().y());
            pt.z = static_cast<float>(point.position().z());
            pt.r = point.color().x();
            pt.g = point.color().y();
            pt.b = point.color().z();
            pointcloud.push_back(pt);
        }

        // add triangles
        pcl::Vertices vertices_idx;
        polygons.reserve(mesh->triangles().size());
        for (const vis::Triangle<vis::u32>& triangle : mesh->triangles()) {
            vertices_idx.vertices.assign({triangle.index(0), triangle.index(1), triangle.index(2)});
            polygons.push_back(vertices_idx);
            vertices_idx.vertices.clear();
        }

        // Converting to the pointcloud binary
        pcl::PCLPointCloud2 pointcloud2;
        pcl::toPCLPointCloud2(pointcloud, pointcloud2);
        // Filling the mesh
        polygon_mesh_ptr->header.frame_id = "world";
        polygon_mesh_ptr->cloud = pointcloud2;
        polygon_mesh_ptr->polygons = polygons;
    }
} //end namespace ROSConversions
#endif //SURFELMESHING_ROS_CONVERSION_TO_ROS_H
