cmake_minimum_required(VERSION 2.8.3)
project(surfelmeshing_ros)

# surfelmeshing actually requires Qt >= 5.8 https://github.com/puzzlepaint/surfelmeshing/issues/10
list(APPEND CMAKE_PREFIX_PATH "/opt/Qt/5.12.5/gcc_64")
# quick fix for weird catkin not found errors in CLion https://github.com/catkin/catkin_tools/pull/391#issuecomment-555285372
list(APPEND CMAKE_PREFIX_PATH "/opt/ros/$ENV{ROS_DISTRO}")
list(APPEND CMAKE_PREFIX_PATH "/opt/ros/$ENV{ROS_DISTRO}/share/catkin/cmake")

enable_language(CUDA)
# required because linking to CUDA libraries from non-.cu files doesn't work with the new enable_language(CUDA) (source: modern-cmake)
find_package(CUDA REQUIRED)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Qt5Widgets REQUIRED)

if(Qt5Widgets_FOUND)
    add_definitions(-DLIBVIS_HAVE_QT)
endif()

find_package(libvis REQUIRED)
#############
# LIBRARIES #
#############

############
# BINARIES #
############

cs_add_executable(surfelmeshing_node
        src/surfelmeshing_node.cpp
        src/surfelmeshing_server.cpp
        src/surfelpipeline.cpp
        src/surfelmeshing_parameters.cpp
        )

target_link_libraries(surfelmeshing_node
        Qt5::Widgets
        ${CUDA_LIBRARIES}
        ${libvis_LIBRARIES})

target_include_directories(surfelmeshing_node PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
##########
# EXPORT #
##########
cs_install()
cs_export()
