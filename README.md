# Ped/Crowd Flow

- python3.6
- [ROS Melodic](http://wiki.ros.org/melodic)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch](http://pytorch.org/)
- [RVO2](https://github.com/sybrenstuvel/Python-RVO2)

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

### Installing RVO2

```
mkdir etc && cd etc
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install Cython
sudo apt-get install cmake
python setup.py build
python setup.py install
```

### Installing tensorboardX(for python 2.7)

```
pip install tensorflow==1.14.0
pip install tensorboardX==1.0
pip install protobuf==3.17.3

tensorboard --logdir runs/
(if you use WSL2, need to install chrome because external access is prohibited, check https://julialang.kr/?p=3181)
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
# TODOList
```
- [x] seperate human and robot policy: generate_action_human(), generate_action_robot() or generate_action_robot_localmap()
- [ ] create flowmap(global/local)
- [x] seperate train / test function for ppo: argument evaluate=True(test), False(train)
- [x] modify model's pth file name from global step to episodes
- [x] change RVO's output velocity(linear) to polinomial as (x,w) -> (x,y)
- [ ] check whether get_laser_observation need -0.5
- [ ] enlarge local map(current 4*4)
- [ ] RVO:make gaussian noise to disable mutation-lock btwn humans  211115
- [ ] visualize LIDAR 1D vector
```
- From IJCAI20 work, crowd-steer
- [ ] [Sim-2-real gap manage] Add Gaussian noise N(0,0.2) to the lidar sensor and goal location
- [ ] [Sim-2-real gap manage] ? Increase the delay in subscribing to current velocity observations to mimic the real-world conditions

'''
- 211105, after meeting
- [ ] visualize local occupancy map(OpenCV)
- [x] prevent human-human collision: modify RVO
- [ ] check all states(lidar, rel.goal,vel) is right
	- 1. stacked lidar data[sensor, sensor, sensor]. mayby sensor L/R sensor issue has
	- 2. relative(local) goal position: 2D vector, goal in polar coordinate(distance and angle) with respect to robot's curr position
	- 3. velocity of robot: current translational and rotational velocity of nonholonomic robot
- [ ] as toward goal, reward is increase well?
- [x] incremental testing: basic scene(0, 1, 2, ... 5): tested scene 0, 5(now)
- [ ] use conv2D, rather than FC for occupancy map
```
# checklist
```
-1. [ppo_city_dense.py]policy_r=RobotPolicy or RobotPolicy_LM
-2. [ppo_city_desne.py] evaluate=true or false (test/training)
```
