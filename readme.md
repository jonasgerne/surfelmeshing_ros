Use with this [fork](https://github.com/jonasgerne/surfelmeshing) to import the SurfelMeshing and libvis libraries via ROS/catkin.

## Build
This guide assumes you have [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html) installed and make sure to set the right paths:
```sh
# make sure the necessary libraries are installed (see surfelmeshing repository)
cd catkin_ws/src
git clone git@github.com:catkin/catkin_simple.git
git clone git@github.com:ethz-asl/eigen_catkin.git
git clone -b ros_interface git@github.com:jonasgerne/surfelmeshing.git
git clone git@github.com:jonasgerne/surfelmeshing_ros.git
# mesh visualization in rviz
git clone git@github.com:uos/mesh_tools.git
# catkin_simple needs to be build first so that the following packages can find it
catkin build catkin_simple eigen_catkin
catkin build libvis -DCATKIN_ENABLE_TESTING=0 -DCATKIN_DEVEL_PREFIX:PATH=$HOME/catkin_ws/devel
catkin build surfelmeshing_ros -DCATKIN_DEVEL_PREFIX:PATH=$HOME/catkin_ws/devel
```

## Run
To run an example with the ROS wrapper, run the following commands. To launch `surfelmeshing_ros` run:
```sh
roslaunch surfelmeshing_ros carla.launch
```
Adjust the launchfile to your personal preferences (image topics, transform topic). 
The node expects a geometry_msgs::TransformStamped message to the camera. 
If the transform points to a different part of the moving object than the camera, 
you can set the parameter "ego_to_cam":
```sh 
<param name="ego_to_cam" value="x y z qx qy qz qw" />
```
### Services
After you received some messages, you can call one of the following services:
```sh 
rosservice call /surfelmeshing/generate_mesh
rosservice call /surfelmeshing/save_ply
```
`generate_mesh` triggers the publishing of the `/surfelmeshing/rosmesh` topic. You 
can view this message in rviz after you build `mesh_tools/rviz_mesh_plugin`.

| Topic                  | Type                          |
|------------------------|-------------------------------|
| /surfelmeshing/rosmesh | mesh_msgs/TriangleMeshStamped |
## Further remarks

I'll try to upload a KITTI or TUM-RGBD example soon.