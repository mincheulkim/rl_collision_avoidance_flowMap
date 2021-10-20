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
"MOST PRIOR TASK"
1. LIDAR t-10, t-5, t
2. visualise LIDAR as 3*512 2D MAP
3. model humans behavior by social force
4. robot only use 1 single mpiexec: human behavior(con_vel) is controlled by other python file, not implementing mpiexecs
   because base human agent already created by groups.world.
5. validate buffer saver
6

```
- [x] seperate human and robot policy: generate_action_human(), generate_action_robot() or generate_action_robot_localmap()
- [x] seperate train / test function for ppo: argument evaluate=True(test), False(train)
- [x] modify model's pth file name from global step to episodes
- [x] change RVO's output velocity(linear) to polinomial as (x,w) -> (x,y)
- [x] check whether get_laser_observation need -0.5: for normalization [-0.5~0.5]
- [x] collision checking module: current ros crash is not perferct. use min(lidar)<robot radius(0.4) to make collision detection
- [ ] enhance collision checking module: make 360degree lidar for collision detection
- [x] Fix segmentation fault issue: in stage_world1.py, there is rospy.sleep() in reset_pose() and reset_world(). make sleep last for a longer, try: set 1.0
- [x] save buffer for next loading: buffer and last_r_v
- 1. Generate Group human
- (Group)Leader - Follower model
-- Leader: Standard RVO velocity
-- Follower: Standard RVO velocity + To-Leader velocity(maintain cohensity) sum vector velocity
- [ ] RVO:make gaussian noise to disable mutation-lock btwn humans  211115
- [ ] after reaching the goal, human change their next goal
- [ ] using social force, create behavior of humans and groups
- [ ] On/Off for visible/invisible robot for human
- 2. Generate Flow Map
- [x] create flowmap(local)    by 211201: ppo.py/generate_action_LM() 6x6 size
- [ ] create flowmap(global as storage)
- [x] enlarge local map(current 4*4): 6*6
- [x] more high resolution local map(current 1m -> 0.1)   so width is 6m(-3~3), total cell is 60x60 (3600 cells).
- [x] visualize local occupancy map(plt.imshow)   by 211201:  but it is too slow
- [x] visualize local occupancy map(OpenCV)  (after 211105 meeting)
- [x] use conv2D, rather than FC for occupancy map: 2 Conv2D, 2 MaxPool2D
- 3. Make subgroups
- [x] gather humans in similar groups
- [ ] DBSCAN or KNN utilize (on-time real group algorithm)
- [ ] regard each groups as fluidic-rigid body, calcurate CoM and nominal velocities, ...
- 4. Reward shaping (after 211217 meeting)
- [ ] future lidar collision penalty linearly(proximity)
- 5. Evaluation Metric
- [ ] most closest(minest) distance with human
- [ ] # of intrusion of personal space
- 6. MISC
- [ ] visualize LIDAR map(OpenCV)  (after 211105 meeting)
- [ ] [Sim-2-real gap manage] Add Gaussian noise N(0,0.2) to the lidar sensor and goal location  (From IJCAI20 work, crowd-steer)
- [ ] [Sim-2-real gap manage] ? Increase the delay in subscribing to current velocity observations to mimic the real-world conditions  (From IJCAI20 work, crowd-steer)
- [ ] change lidar timestamp t-2, t-1, t -> t-10, t-5, t
- [ ] change python 3 environment (for social force)

'''
- 211105, after meeting
- [x] prevent human-human collision: modify RVO
- [x] check all states(lidar, rel.goal,vel) is right
	- 1. stacked lidar data[sensor, sensor, sensor]. mayby sensor L/R sensor issue has
	- 2. relative(local) goal position: 2D vector, goal in polar coordinate(distance and angle) with respect to robot's curr position
	- 3. velocity of robot: current translational and rotational velocity of nonholonomic robot
- [x] as toward goal, reward is increase well?
- [x] incremental testing: basic scene(0, 1, 2, ... 5): tested scene 0, 5(now)

```
# checklist
```
-1. [ppo_city_dense.py]policy_r=RobotPolicy or RobotPolicy_LM
-2. [ppo_city_desne.py] evaluate=true or false (test/training)
```

# Motivation
'''
--- Incorporate lidar based and state based
1. If we use only lidar data, 1) we cannot identify static obstacle and dynamic obstacle 2) loose high quality of intented human's information
2. If we use only state based, 1) it is unrelistic that we assumed know all state of humans 2) scalability issue
--- especially, dence scenario, above problem increased
--- so, we utilize identified information of humans to make rigid-body movement or flow, to see high-level instance
'''