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
import message_filters




class StageWorld():
    def __init__(self, beam_num, index, num_env):    # called from ppo_stage3.py,   # 512, index, 5
        self.index = index
        self.num_env = num_env   # 5 agents
        node_name = 'StageEnv_' + str(index)   # stageEnv_0?
        rospy.init_node(node_name, anonymous=None)
        
        

        # Define Subscriber

        sub_list = []          # https://velog.io/@suasue/Python-%EA%B0%80%EB%B3%80%EC%9D%B8%EC%9E%90args%EC%99%80-%ED%82%A4%EC%9B%8C%EB%93%9C-%EA%B0%80%EB%B3%80%EC%9D%B8%EC%9E%90kwargs
        for i in range(11):
            sub = message_filters.Subscriber('robot_' + str(i) + '/base_pose_ground_truth', Odometry)
            sub_list.append(sub)
        #print('sub_list=',sub_list)

        queue_size = 10
        fps = 1000.
        delay = 1 / fps * 0.5

        #mf = message_filters.ApproximateTimeSynchronizer([sub1, sub2], queue_size, delay)
        mf = message_filters.ApproximateTimeSynchronizer(sub_list, queue_size, delay)
        mf.registerCallback(self.callback)

        self.pose_list = []

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

        self.flow_map = None  # 211026

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
        self.speed_poly = None  # 211103
        self.is_crashed = None

        # TODO for lidar collision check 211130
        self.is_collision = 0
        self.scan_min = 6.0


        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None or self.speed_poly is None:
            pass
        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)

    def callback(self, *msgs):
        pose_list = []
        
        #self.pose_list=[]
        
        for msg in msgs:
            #msg_list.append(i)
            Quaternious = msg.pose.pose.orientation
            Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
            #self.state_GT = [msg.pose.pose.position.x, msg.pose.pose.position.y, Euler[2]]
            x,y=msg.pose.pose.position.x, msg.pose.pose.position.y
            pose_list.append([x,y, Euler[2]])
            #self.pose_list.append([x,y, Euler[2]])
        self.pose_list = pose_list
        
        
  
        #self.post_list = post_lists
        #for i in range(len(msg_list)):
        #    x,y=msg_list[i].pose.pose.position.x, msg_list[i].pose.pose.position.y
        #    pose_list.append([x,y])
        #print('pose_list:',pose_list)
        
    def ground_truth_callback(self, GT_odometry):   # topic: robot_0_base_pose_ground topic callback F
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]
        self.speed_poly = [v_x, v_y]


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
    
    def get_env_pose(self):
        return self.pose_list

    def get_self_speed_poly(self):
        return self.speed_poly

    def get_self_state_rot(self):
        return self.state_GT[2]

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)  # from laser_scan_callback   # self.scan = np.array(scan.ranges)
        scan[np.isnan(scan)] = 6.0       # NaN or INF set 6
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum   # 512
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []   # left scan
        index = 0.

        for x in range(int(sparse_beam_num / 2)):   # routine 256
            sparse_scan_left.append(scan[int(index)])
            index += step

        sparse_scan_right = []   # right scan
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)   # concat left, right scan(flip)
        #scan_sparse = np.flip(scan_sparse)    # 211115
        scan_sparse = scan_sparse[::-1]    # 211115  fliped input
        return scan_sparse / 6.0 - 0.5   # because sensor are front of robot(50cm)

    def collision_laser_flag(self, r):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0

        scan_min = np.min(scan)

        if scan_min <= r:
            self.is_collision = 1
        else:
            self.is_collision = 0

        self.scan_min = scan_min


    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed   # from ROS callback F is_crashed int(1 or 0)

    def get_sim_time(self):
        return self.sim_time

    def get_goal_point(self):  # 211102
        return self.goal_point

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()   # robot state
        [goal_x, goal_y] = self.goal_point        # robot generated goal(global)
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)   # relative robot aspect to goal(local goal)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        #rospy.sleep(0.5)
        rospy.sleep(1.0)   # 211214

    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()   # generate goal 1) dist to zero > 9, 2) 8<dist to agent<10
        self.goal_point = [x_g, y_g]                 # set "global" goal
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)

    # TODO: Reward reshape: penalty for circuling around
    def get_reward_and_terminate(self, t, scaled_action):   # t is increased 1, but initializezd 1 when terminate=True
        terminate = False
        laser_scan = self.get_laser_observation()   # new laser scan(Because excuted action)
        [x, y, theta] = self.get_self_stateGT()     # "updated" current state
        [v, w] = self.get_self_speedGT()            # updated current velocity

        self.pre_distance = copy.deepcopy(self.distance)   # previous distance to local goal
        # Propotional Reward
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)  # updated new distance to local goal after action
        reward_g = (self.pre_distance - self.distance) * 2.5  # REWARD for moving forward, later reach goal reward(+15)  # original
        #reward_g = (self.pre_distance - self.distance) * 1.5  # REWARD for moving forward, later reach goal reward(+15)
        #if reward_g<0:
        #    reward_g =0
        reward_c = 0  # collision penalty
        reward_w = 0  # too much rotation penalty
        result = 0
        is_crash = self.get_crash_state()   # return self.is_crashed

        if self.distance < self.goal_size:  # success reward
            terminate = True
            #reward_g = 15
            reward_g = 35
            result = 'Reach Goal'

        if self.index==0:
            if is_crash == 1:                   # collision penalty
                terminate = True
                reward_c = -15.
                #reward_c = -30.
                result = 'Crashed'

        # TODO Lidar collsion check 211215
        self.collision_laser_flag(r=0.4)
        if self.index == 0 and self.is_collision==1:
            #print(self.index, 'is_collision : ',self.is_collision, 'min_dist_LIDAR:',self.scan_min)
        #if self.is_collision == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        
        if np.abs(w) >  1.05:               # rotation penalty
            reward_w = -0.1 * np.abs(w)
            #reward_w = -0.45 * np.abs(w) **2
        '''
        if np.abs(scaled_action[1])>1.05:
            reward_w = -0.45 * np.abs(scaled_action[1])**2    # 211221 from DSRNN
        '''

        # 211221 add a penalty for going backwoards
        if scaled_action[0]<0:
            r_back = -0.45 * np.abs(scaled_action[0])
        else:
            r_back = 0
        #if self.index==0:
        #        print('v:',scaled_action[0],'w:',scaled_action[1],'rot_penalty:',reward_w,'r_back:',r_back)

        #if t > 150:  # timeout check
        if t > 1000:  # timeout check  211020 for long-term
            terminate = True      
            result = 'Time out'
        
        '''    # for future lidar collision penalty
        if (self.scan_min > self.robot_radius[0]) and (self.scan_min < (self.lidar_danger+self.robot_radius[0])):
            reward_ct = -0.25*((self.lidar_danger+self.robot_radius[0]) - self.scan_min)
        '''            
        
        # 211221 for unicycle model, backward penalty
        reward = reward_g + reward_c + reward_w
        #reward = reward_g + reward_c + reward_w + r_back # 211221
        #if self.index==0:
        #    print(reward, 'reward_g:',reward_g, 'reward_b:',r_back)
        '''
        else:  # TODO time penalty
            reward_t = -0.1
        '''

        return reward, terminate, result   # float, T or F(base), description

    def reset_pose(self):
        random_pose = self.generate_random_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9     # this lines are for random start pose
        #random_pose = self.generate_random_circle_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9     # this lines are for random start pose
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
        
    def control_vel_specific(self, action, index):   # real action as array[0.123023, -0.242424]. from
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        cmd_vel.publish(move_cmd)
        
    def control_vel_poly(self, action,diff):   # real action as array[0.123023, -0.242424]. from
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = action[1]
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = diff
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
        
    def control_pose_specific(self, pose, index):    # pose = [x, y, theta]
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
        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)
        
        cmd_pose.publish(pose_cmd)

    def generate_random_pose(self):
        #x = np.random.uniform(-9, 9)
        #y = np.random.uniform(-9, 9)
        # for robot(agent0)
        if self.index == 0:
            x = 0
            y = -8
        # For human
        elif self.index == 1:
            x=-6
            y=-6
        elif self.index == 2:
            x= 6
            y=-6
            #x,y=-17,-17
        elif self.index == 3:
            x= 6
            y= 6
        elif self.index == 4:
            x= -6
            y= 6
        else:
            x = np.random.uniform(-7, 7)
            y = np.random.uniform(-7, 7)
        #dis = np.sqrt(x ** 2 + y ** 2)
        #while (dis > 9) and not rospy.is_shutdown():
        #    x = np.random.uniform(-9, 9)
        #    y = np.random.uniform(-9, 9)
        #    dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(0, 0.5 * np.pi)
        #if self.index ==0:
        #    theta = np.pi*0.5
        return [x, y, theta]

    # 211110
    def generate_random_circle_pose(self):
        if self.index == 0:   # for robot
            x = 0
            y = -8
        else:           # For human
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-8, 8)
            dis = np.sqrt(x ** 2 + y ** 2)
            while (dis < 7.5 or dis > 8.5) and not rospy.is_shutdown():
                x = np.random.uniform(-8, 8)
                y = np.random.uniform(-8, 8)
                dis = np.sqrt(x ** 2 + y ** 2)
        # for test city scene, use below
        '''
        else:
            x=self.index - 10
            y=self.index - 10
        '''

        #theta = np.random.uniform(0, 0.5 * np.pi)
        #theta = np
        if self.index ==0:
            theta = np.random.uniform(0, 2*np.pi)
        else:
            theta = np.arctan2(y, x) + np.pi
        #if self.index ==0:
        #    theta = np.pi*0.5
        return [x, y, theta]

    # 211129
    def generate_group_pose(self):
        if self.index == 0:   # for robot
            x = 0
            y = -8
        elif self.index in [1,2,3]:
            x = np.random.uniform(-7.5, -5.5)
            y = np.random.uniform(-7.5, -5.5)
        elif self.index in [4,5]:
            x = np.random.uniform(-7.5, -5.5)
            y = np.random.uniform(5.5, 7.5)
        elif self.index in [6]:
            x = np.random.uniform(6, 7)
            y = np.random.uniform(6, 7)
        else:           # For human
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-8, 8)
            dis = np.sqrt(x ** 2 + y ** 2)
            while (dis < 7.5 or dis > 8.5) and not rospy.is_shutdown():
                x = np.random.uniform(-8, 8)
                y = np.random.uniform(-8, 8)
                dis = np.sqrt(x ** 2 + y ** 2)

        #theta = np.random.uniform(0, 0.5 * np.pi)
        #theta = np
        if self.index ==0:
            theta = np.random.uniform(0, 2*np.pi)
        else:
            theta = np.arctan2(y, x) + np.pi
        #if self.index ==0:
        #    theta = np.pi*0.5
        return [x, y, theta]

    def generate_random_goal(self):
        self.init_pose = self.get_self_stateGT()
        # For robot
        if self.index == 0:
            x = 0
            y = 8
        # For human
        elif self.index == 1:
            x= 6
            y= 6
            #x,y=-18,-18
        elif self.index == 2:
            x= -6
            y=6
        elif self.index == 3:
            x= -6
            y= -6
        elif self.index == 4:
            x= 6
            y= -6
        else:
            x = np.random.uniform(-7, 7)
            y = np.random.uniform(-7, 7)
        #x = np.random.uniform(6, 9)   314    182  5
        #y = np.random.uniform(6, 9)
        dis_origin = np.sqrt(x ** 2 + y ** 2)
        dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)
        #while (dis_origin > 9 or dis_goal > 10 or dis_goal < 8) and not rospy.is_shutdown():
        #    x = np.random.uniform(-9, 9)
        #    y = np.random.uniform(-9, 9)
        #    dis_origin = np.sqrt(x ** 2 + y ** 2)
        #    dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)

        return [x, y]

    def generate_pose_goal_circle(self):
        # reset pose
        #random_pose = self.generate_random_circle_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9     # this lines are for random start pose
        random_pose = self.generate_group_pose()   # 211129. Groups initialize
        #rospy.sleep(0.01)
        rospy.sleep(1.0)   # too laggy
        self.control_pose(random_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:  # np.bas: absolute, compare # generated random pose with topic pose
            [x_robot, y_robot, theta] = self.get_self_stateGT()    # same
            self.control_pose(random_pose)

        #rospy.sleep(0.01)
        rospy.sleep(1.0)   # too laggy
        self.is_crashed=False

        # reset goal
        #[x_g, y_g] = self.generate_random_goal()   # generate goal 1) dist to zero > 9, 2) 8<dist to agent<10
        self.init_pose = self.get_self_stateGT()
        if self.index == 0:
            # for cross scene
            self.goal_point = [0, 8]
            # for city scene
            #self.goal_point = [-13, 10]
        elif self.index in [1,2,3]:
            self.goal_point = [8.0, 0.0]
        elif self.index in [4,5]:
            self.goal_point = [0.0, -8.0]
        elif self.index in [6]:
            self.goal_point = [-8.0, 8.0]
        else:
            self.goal_point = [-random_pose[0], -random_pose[1]]                 # set "global" goal
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)
        
        
    def initialize_pose_robot_humans(self):   # 211222
        # reset pose
        #random_pose = self.generate_random_circle_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9     # this lines are for random start pose
        '''
        random_pose = self.generate_group_pose()   # 211129. Groups initialize
        #rospy.sleep(0.01)
        rospy.sleep(1.0)   # too laggy
        self.control_pose(random_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:  # np.bas: absolute, compare # generated random pose with topic pose
            [x_robot, y_robot, theta] = self.get_self_stateGT()    # same
            self.control_pose(random_pose)

        #rospy.sleep(0.01)
        rospy.sleep(1.0)   # too laggy
        self.is_crashed=False

        # reset goal
        #[x_g, y_g] = self.generate_random_goal()   # generate goal 1) dist to zero > 9, 2) 8<dist to agent<10
        self.init_pose = self.get_self_stateGT()
        if self.index == 0:
            # for cross scene
            self.goal_point = [0, 8]
            # for city scene
            #self.goal_point = [-13, 10]
        elif self.index in [1,2,3]:
            self.goal_point = [8.0, 0.0]
        elif self.index in [4,5]:
            self.goal_point = [0.0, -8.0]
        elif self.index in [6]:
            self.goal_point = [-8.0, 8.0]
        else:
            self.goal_point = [-random_pose[0], -random_pose[1]]                 # set "global" goal
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)
        '''
        
        '''
        return [[0,-8,0.785],
                [-5,-5,0.785],[-6,-6,0.785],[-6,-5,0.785],[-5,-6,0.785],[-7,-6,0.785],
                [-5,5,0.785],[-5,6,0.785],[-6,5,0.785],
                [7,1,0.785*4],[7,2,0.785*4]],[
                    [0,8],
                    [5,5],[6,6],[6,5],[5,6],[7,6],
                    [5,-5],[5,-6],[6,-5],
                    [-7,-1],[-7,-2]]
        '''
        total_list = []
        '''
        for index in range(11):
            object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
            self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)    # input stage(argument), dataType, called function
            total_list.append(self.state_GT)
        
        print(total_list)
        
        '''
        return [[0,-8,np.pi/2],
                    [-7,-0,np.pi/2],[-6,-0,np.pi/2],[-5,-0,np.pi/2],[-4,-0,np.pi/2],[-3,-0,np.pi/2],
                    [-2,0,np.pi*3/2],[-1,0,np.pi*3/2],[-0,0,np.pi*3/2],
                    [1,0,np.pi*3/2],[2,0,np.pi*3/2]],[
                [0,8],
                    [-2,7],[-1,7],[-0,7],[1,7],[4,7],
                    [-2,-7],[-1,-7],[-0,-7],
                    [1,-7],[2,-7]]
        
        
    def set_init_pose(self, init_pose):
        self.control_pose(init_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose
        rospy.sleep(0.01)
        
    
    def set_init_goal(self, init_goal):
        self.goal_point = [init_goal[0],init_goal[1]]                 
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)