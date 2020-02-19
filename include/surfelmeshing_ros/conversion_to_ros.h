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

namespace ROSConversions{
    inline void storeToMeshHelper(mesh_msgs::TriangleMesh& mesh_msg, const vis::PointCloud<vis::Point3f>& cloud) { //3fCloud& cloud) {
        for (vis::usize i = 0; i < cloud.size(); ++ i) {
            const vis::Point3f& point = cloud.data()[i];
            geometry_msgs::Point point_msg;
            point_msg.x = point.position().x();
            point_msg.y = point.position().y();
            point_msg.z = point.position().z();
            mesh_msg.vertices.push_back(point_msg);
        }
    }

    inline void storeToMeshHelper(mesh_msgs::TriangleMesh& mesh_msg, const vis::PointCloud<vis::Point3fCu8>& cloud) {
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

    inline void storeToMeshHelper(mesh_msgs::TriangleMesh& mesh_msg, const vis::PointCloud<vis::Point3fC3u8>& cloud) {
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

    inline void storeToMeshHelper(mesh_msgs::TriangleMesh& mesh_msg, const vis::PointCloud<vis::Point3fC3u8Nf>& cloud) {
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

        storeToMeshHelper(mesh_msg, (*mesh->vertices()));
        mesh_msg.triangles.reserve(num_points/3);

        for (const vis::Triangle<vis::u32>& triangle : mesh->triangles()){
            mesh_msgs::TriangleIndices indices_msg;
            for (int j = 0; j < 3; j++)
                indices_msg.vertex_indices[j] = static_cast<uint32_t>(triangle.index(j));
            mesh_msg.triangles.push_back(indices_msg);
        }

        mesh_msg_stmpd->mesh = mesh_msg;
    }

} //end namespace ROSConversions
#endif //SURFELMESHING_ROS_CONVERSION_TO_ROS_H
