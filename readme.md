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
git clone git@github.com:uos/mesh_tools.git
# catkin_simple needs to be build first so that the following packages can find it
catkin build catkin_simple eigen_catkin
catkin build libvis -DCATKIN_ENABLE_TESTING=0 -DCATKIN_DEVEL_PREFIX:PATH=$HOME/catkin_ws/devel
catkin build surfelmeshing_ros -DCATKIN_DEVEL_PREFIX:PATH=$HOME/catkin_ws/devel
```