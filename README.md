# Ped/Crowd Flow

- python3.6
- [ROS Melodic](http://wiki.ros.org/melodic)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch](http://pytorch.org/)

## setup

```
$ mkdir PROJECT_WS/src
```

### Using python3 and tf in ROS, Stage-ros
```
$ virtualenv -p /usr/bin/python3 [VENV_PATH]
$ source [VENV_PATH]/bin/activate
$ pip install catkin_pkg pyyaml empy rospkg numpy
$ cd PROJECT_WS/src
$ cp stage_ros-add_pose_and_crash PROJECT_WS/src
$ git clone --branch melodic-devel https://github.com/ros/geometry.git
$ git clone --branch melodic-devel https://github.com/ros/geometry2.git
$ cd ..
$ catkin_make -DPYTHON_EXECUTABLE:FILEPATH=[VENV_PYTHON_PATH]
$ source devel/setup.bash
```

Note that you can find your own "VENV_PYTHON_PATH" using the following commands.
```
$ source [VENV_PATH]/bin/activate
$ which python
```



## References

```
@misc{Tianyu2018,
	author = {Tianyu Liu},
	title = {Robot Collision Avoidance via Deep Reinforcement Learning},
	year = {2018},
	publisher = {GitHub},
	journal = {GitHub repository},
	howpublished = {\url{https://github.com/Acmece/rl-collision-avoidance.git}},
	commit = {7bc682403cb9a327377481be1f110debc16babbd}
}
```
# rl_collision_avoidance_flowMap
