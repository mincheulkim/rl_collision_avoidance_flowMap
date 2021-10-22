import time
import rospy
import copy
import tf
import numpy as np


from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
from model.utils import test_init_pose_new, test_goal_point_new    # arbitrary start and goal position sets



class StageWorld():
    def __init__(self, beam_num, index, num_env):    # called from ppo_stage3.py,   # 512, index, 5
        self.index = index
        self.num_env = num_env   # 5 agents
        node_name = 'StageEnv_' + str(index)   # stageEnv_0?
        rospy.init_node(node_name, anonymous=None)

        self.beam_mum = beam_num   # 512
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.
        # self.reset_pose = None

        self.init_pose = None



        # for get reward and terminate(Didn't use)
        self.stop_counter = 0

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)    # input stage(argument), dataType, called function

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)


        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            pass

        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)


    def ground_truth_callback(self, GT_odometry):   # topic: robot_0_base_pose_ground topic callback F
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)  # from laser_scan_callback   # self.scan = np.array(scan.ranges)
        scan[np.isnan(scan)] = 6.0       # NaN or INF set 6
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum   # 512
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []   # left scan
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):   # routine 256
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []   # right scan
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)   # concat left, right scan
        #print('laser scan: ',scan_sparse / 6.0 - 0.5)
        return scan_sparse / 6.0 - 0.5


    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed   # from ROS callback F is_crashed int(1 or 0)

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()   # robot state
        [goal_x, goal_y] = self.goal_point        # robot generated goal
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)   # relative robot aspect to goal(local goal)
        return [local_x, local_y]
        #return [goal_x, goal_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)


    def generate_goal_point(self):
        '''
        [x_g, y_g] = test_goal_point_new(self.index)   # get index's goal_point
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()
        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)
        '''
        [x_g, y_g] = self.generate_random_goal()   # generate goal 1) dist to zero > 9, 2) 8<dist to agent<10
        #[x_g, y_g] = [0, 0]
        self.goal_point = [x_g, y_g]                 # set global goal
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)
        
        


    def get_reward_and_terminate(self, t):   # t is increased 1, but initializezd 1 when terminate=True
        terminate = False
        laser_scan = self.get_laser_observation()   # new laser scan(Because excuted action)
        [x, y, theta] = self.get_self_stateGT()     # "updated" current state
        [v, w] = self.get_self_speedGT()            # updated current velocity
        self.pre_distance = copy.deepcopy(self.distance)   # previous distance to local goal
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)  # updated new distance to local goal after action
        reward_g = (self.pre_distance - self.distance) * 2.5  # REWARD for moving forward, later reach goal reward(+15)
        reward_c = 0  # collision penalty
        reward_w = 0  # too much rotation penalty
        result = 0
        is_crash = self.get_crash_state()   # return self.is_crashed

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        if np.abs(w) >  1.05:   # rotation penalty
            reward_w = -0.1 * np.abs(w)

        #if t > 150:  # timeout check
        if t > 450:  # timeout check  211020 for long-term
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result   # float, T or F(base), description

    def reset_pose(self):
        
        '''
        reset_pose = test_init_pose_new(self.index)   # from model.utils
        self.control_pose(reset_pose)                 # if want static start pose, use above two lines 211021
        '''
        random_pose = self.generate_random_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9     # this lines are for random start pose
        rospy.sleep(0.01)
        self.control_pose(random_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:  # np.bas: absolute, compare # generated random pose with topic pose
            [x_robot, y_robot, theta] = self.get_self_stateGT()    # same
            self.control_pose(random_pose)
        
        rospy.sleep(0.01)


    def control_vel(self, action):   # real action as array[0.123023, -0.242424]. from 
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)


    def control_pose(self, pose):    # pose = [x, y, theta]
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]   # x
        pose_cmd.position.y = pose[1]   # y
        pose_cmd.position.z = 0         # 0(don't care rot cause pose?)

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

    def generate_random_pose(self):
        #x = np.random.uniform(-9, 9)
        #y = np.random.uniform(-9, 9)
        if self.index > 14:
            x = np.random.uniform(-7, -9)
            y = np.random.uniform(7, 9)
        elif self.index > 9:
            x = np.random.uniform(7, 9)
            y = np.random.uniform(7, 9)
        elif self.index > 4:
            x = np.random.uniform(7, 9)
            y = np.random.uniform(-7, -9)
        else:
            x = np.random.uniform(-9, -7)
            y = np.random.uniform(-9, -7)
        
        #dis = np.sqrt(x ** 2 + y ** 2)
        #while (dis > 9) and not rospy.is_shutdown():
        #    x = np.random.uniform(-9, 9)
        #    y = np.random.uniform(-9, 9)
        #    dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(0, 0.5 * np.pi)
        return [x, y, theta]

    def generate_random_goal(self):
        self.init_pose = self.get_self_stateGT()
        if self.index > 14:
            x = np.random.uniform(-7, -9)
            y = np.random.uniform(-7, -9)
        elif self.index > 9:
            x = np.random.uniform(-7, -9)
            y = np.random.uniform(7, 9)
        elif self.index > 4:
            x = np.random.uniform(7, 9)
            y = np.random.uniform(7, 9)
        else:
            x = np.random.uniform(7, 9)
            y = np.random.uniform(-7, -9)
        #x = np.random.uniform(6, 9)
        #y = np.random.uniform(6, 9)
        #x = np.random.uniform(6, 7)
        #y = np.random.uniform(7, 8)
        #x = -4
        #y = 8
        dis_origin = np.sqrt(x ** 2 + y ** 2)
        dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)
        #while (dis_origin > 9 or dis_goal > 10 or dis_goal < 8) and not rospy.is_shutdown():
        #    x = np.random.uniform(-9, 9)
        #    y = np.random.uniform(-9, 9)
        #    dis_origin = np.sqrt(x ** 2 + y ** 2)
        #    dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)

        return [x, y]





