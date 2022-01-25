import time
import rospy
import copy
import tf
import numpy as np


import lidar_to_grid_map as lg
import matplotlib.pyplot as plt
import cv2


from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
import message_filters

from scipy.spatial import ConvexHull

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal




class StageWorld():
    def __init__(self, beam_num, index, num_env):    # called from ppo_stage3.py,   # 512, index, 5
        self.index = index
        self.num_env = num_env   # 5 agents
        node_name = 'StageEnv_' + str(index)   # stageEnv_0?
        rospy.init_node(node_name, anonymous=None)

        self.pose_list = []
        self.speed_poly_list = []  # 220104
        
        self.crash_list = []  # 220107

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
        
        # Initialize Groups and Humans  # 220103
        self.num_human = 6        # VERY SIMPLE SCENE (5 circle cross individual)
        #self.num_human = 11
        #self.num_human = 14      # 
        #self.num_human = 22     # 220111(+14)
        
        
        self.groups = [0, 1, 2, 3, 4, 5]
        #self.groups = [0, 1, 2, 3]
        #self.groups = [0, 1, 2, 3, 4]       # 220110
        #self.groups = [0, 1, 2, 3, 4, 5]   # 220111
        
        
        self.human_list=[[0],[1],[2],[3],[4],[5]]
        #self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10]]
        #self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10],[11,12,13]]                               # 220110
        #self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10],[11,12,13], [14,15,16,17,18,19,20,21]]   # 220111
        
        # Define Subscriber
        sub_list = []          # https://velog.io/@suasue/Python-%EA%B0%80%EB%B3%80%EC%9D%B8%EC%9E%90args%EC%99%80-%ED%82%A4%EC%9B%8C%EB%93%9C-%EA%B0%80%EB%B3%80%EC%9D%B8%EC%9E%90kwargs
        sub_crash_list = []
        
        for i in range(self.num_human):
        #for i in range(21):   # 220102
            sub = message_filters.Subscriber('robot_' + str(i) + '/base_pose_ground_truth', Odometry)
            sub_list.append(sub)
            
            crash_sub = message_filters.Subscriber('robot_' + str(i) + '/is_crashed', Int8)
            sub_crash_list.append(crash_sub)
            
        queue_size = 10
        fps = 1000.
        delay = 1 / fps * 0.5

        mf = message_filters.ApproximateTimeSynchronizer(sub_list, queue_size, delay)
        mf.registerCallback(self.callback)
        
        crash_mf = message_filters.ApproximateTimeSynchronizer(sub_crash_list, queue_size, delay, allow_headerless=True)
        crash_mf.registerCallback(self.crash_callback_mf)


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
        
        self.map1 = None

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
        speed_poly_list=[]
        
        for msg in msgs:
            #msg_list.append(i)
            Quaternious = msg.pose.pose.orientation
            Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
            #self.state_GT = [msg.pose.pose.position.x, msg.pose.pose.position.y, Euler[2]]
            x,y=msg.pose.pose.position.x, msg.pose.pose.position.y
            vx=msg.twist.twist.linear.x
            vy=msg.twist.twist.linear.y
            pose_list.append([x,y, Euler[2]])
            speed_poly_list.append([vx,vy])
            
            #self.pose_list.append([x,y, Euler[2]])
        self.pose_list = pose_list
        self.speed_poly_list = speed_poly_list
        
        #self.post_list = post_lists
        #for i in range(len(msg_list)):
        #    x,y=msg_list[i].pose.pose.position.x, msg_list[i].pose.pose.position.y
        #    pose_list.append([x,y])
        #print('pose_list:',pose_list)
        
    def crash_callback_mf(self, *msgs):
        crash = []
        
        for msg in msgs:
            crash.append(msg.data)
            
        self.crash_list = crash
        
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
        step = float(raw_beam_num) / sparse_beam_num   # 1.0
        sparse_scan_left = []   # left scan
        index = 0.
        
        # 220124 create 2D Grid map(3x3, forward of robot, back is dark) from lidar_sensor
        [x, y, theta] = self.get_self_stateGT()     # robot's pose
        map1 = np.ones((60, 60)) * 0.0   # 60 by 60 with default value as 0.5
        inc_angle = np.pi/sparse_beam_num     # np.pi(3.14=180deg) / 512 = 0.00625deg...
        end_list = []
        for i in range(sparse_beam_num):
            distance = scan[int(i)]
            end_list.append([x+distance*np.cos(inc_angle*i),y+distance*np.sin(inc_angle*i)])
            mod_x = np.floor((3+distance*np.cos(inc_angle*i))/0.1)
            mod_y = np.ceil((3-distance*np.sin(inc_angle*i))/0.1)

            if mod_x>59:
                mod_x=59
            if mod_y>59:
                mod_y=59
            if mod_y<0:
                mod_y=0
            if mod_x<0:
                mod_x=0

            line = lg.bresenham((30, 30), (int(mod_y),int(mod_x)))
            for l in line:
                #self.map1[l[0]][l[1]] = 0.5
                map1[l[0]][l[1]] = 0.5
                
        self.map1 = map1
        
        '''
        # Visualize
        hsv=cv2.resize(self.map1, dsize=(480,480), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image',hsv)
        cv2.waitKey(1)
        '''
        

        for x in range(int(sparse_beam_num / 2)):   # 256
            sparse_scan_left.append(scan[int(index)])
            index += step    # 1~256


        sparse_scan_right = []   # right scan
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step    # 510~255
            #print(index, step)

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
        
    def get_min_lidar_dist(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        scan_min=np.min(scan)
        return scan_min
        
    


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
    
    def get_sensor_map(self):
        return self.map1


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
    def get_reward_and_terminate(self, t, scaled_action, policy_list):   # t is increased 1, but initializezd 1 when terminate=True
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
            reward_g = 15
            #reward_g = 35
            result = 'Reach Goal'

        if is_crash == 1:                   # collision penalty
            terminate = True
            reward_c = -15.
            #reward_c = -30.
            result = 'Crashed(ROS)'
        
        min_dist_rrr = 10.0   # 220119
        # 220119 Collision check by rel.dist around 360 degree
        pose_list_np = np.array(self.pose_list)
        rel_dist_list = pose_list_np[:,0:2]-pose_list_np[0,0:2]
        '''
        # 220124 disabled for 더 높은 성공률 위해
        for i in rel_dist_list[1:]:
            min_dist = np.sqrt(i[0]**2+i[1]**2)
            if min_dist < min_dist_rrr:
                min_dist_rrr = min_dist
            if min_dist < 0.6:
                terminate = True
                reward_c = -15.
                result = 'Crashed(Compuatation)'
                break
        '''
            
        # 220119. 관측된 라이다 거리에 반비례해서 penalty linear하게 받게. for 충돌 회피. 1 = 0, 0.8 = 0.2, 0.6 = 0.4
        kkk = self.get_min_lidar_dist()
        #print(kkk, min_dist_rrr)
        penalty_lidar = 0.
        if kkk <= 1.0:
            penalty_lidar = (-1. + kkk)/10
        
        # 220119. timestep penalty
        #constant_penalty = -0.005
        
        '''
        # Lidar collsion check 211215
        self.collision_laser_flag(r=0.4)
        if self.is_collision==1:
            #print(self.index, 'is_collision : ',self.is_collision, 'min_dist_LIDAR:',self.scan_min)
            terminate = True
            reward_c = -15.
            result = 'Crashed(LIDAR)'
        '''
        
        if np.abs(w) >  1.05:               # rotation penalty
            reward_w = -0.1 * np.abs(w)
            #reward_w = -0.45 * np.abs(w) **2

        # 211221 add a penalty for going backwoards
        if scaled_action[0]<0:
            r_back = -0.45 * np.abs(scaled_action[0])
        else:
            r_back = 0

        if t > 1000:  # timeout check  220119 after weekly. our가 TO 더 높게 나와서, 더 크게 줌
        #if t > 700:  # 220107
            terminate = True      
            result = 'Time out'
        
        '''    # for future lidar collision penalty
        if (self.scan_min > self.robot_radius[0]) and (self.scan_min < (self.lidar_danger+self.robot_radius[0])):
            reward_ct = -0.25*((self.lidar_danger+self.robot_radius[0]) - self.scan_min)
        '''            
        
        #print(idx, g_cluster[0], g_cluster)
        # 211231 compute distance from robot to convex hull of groups
        '''
        dist_to_grp = np.zeros(len(g_cluster))
        for j in range(len(g_cluster)):
            #print(g_cluster[j], len(g_cluster[j][0]))
            if len(g_cluster[j][0])==0 or len(g_cluster[j][0])==1:
                print('zero or soely!')
            elif len(g_cluster[j][0])==2:  # 2 humans                
                dist_to_grp[j] = self.point_to_segment_dist(g_cluster[j][0][0][0],g_cluster[j][0][0][1],g_cluster[j][0][1][0],g_cluster[j][0][1][1], x, y)
            else:       # more than 3
                hull = ConvexHull(g_cluster[j][0])
                dists = []
                vert_pos = g_cluster[j][0][hull.vertices]
                for i in range(len(vert_pos)-1):
                    dists.append(self.point_to_segment_dist(vert_pos[i][0],vert_pos[i][1],vert_pos[i+1][0],vert_pos[i+1][1], x, y))
                dist_to_grp[j] = min(dists)
        collision_dist = 0.5
        coll_grp = np.array([1 if (dist_to_grp[j] < collision_dist) else 0 for j in range(len(g_cluster))])
        reward_grp = -0.25 * coll_grp.sum()   # 0,1,2,~ max grp num
        '''
        
        # 220124
        if policy_list == 'concat_LM' or policy_list =='stacked_LM' or policy_list=='LM' or policy_list=='depth_LM' or policy_list=='baseline_LM':
            reward = reward_g + reward_c + reward_w + penalty_lidar # 220119 관측된 lidar dist 비례 페널티 추가
        elif policy_list == '':
            reward = reward_g + reward_c + reward_w
        #reward = reward_g + reward_c + reward_w  # original befrom 220119
        #reward = reward_g + reward_c + reward_w + reward_grp  # 211231 dynamic group collision penalty added
        #reward = reward_g + reward_c + reward_w + r_back # 211221   # raw policy에서 뒤로 갈때 페널티
        '''
        else:  # TODO time penalty
            reward_t = -0.1
        '''
        #print('reward:',reward)
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
        if self.index == 0:  # For robot
            x = 0
            y = -8
        elif self.index == 1:    # For human
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

        #theta = np.random.uniform(0, 0.5 * np.pi)
        #theta = np
        if self.index ==0:
            theta = np.random.uniform(0, 2*np.pi)
        else:
            theta = np.arctan2(y, x) + np.pi
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

        if self.index ==0:
            theta = np.random.uniform(0, 2*np.pi)
        else:
            theta = np.arctan2(y, x) + np.pi
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
        
        
    def initialize_pose_robot_humans(self, rule):   # 211222        
        init_pose_list = [[0,-8,np.pi/2],
                    [-7,-0,np.pi/2],[-6,-0,np.pi/2],[-5,-0,np.pi/2],[-4,-0,np.pi/2],[-3,-0,np.pi/2],
                    [-2,0,np.pi*3/2],[-1,0,np.pi*3/2],[-0,0,np.pi*3/2],
                    [1,0,np.pi*3/2],[2,0,np.pi*3/2]]
        init_goal_list = [[0,8],
                    [-2,7],[-1,7],[-0,7],[1,7],[4,7],
                    [-2,-7],[-1,-7],[-0,-7],
                    [1,-7],[2,-7]]

        if rule=='group_circle_crossing':   #         Rule circle_crossing: generate start position on a circle, goal position is at the opposite side
            #print('scenario:',rule)
            init_pose_list=[]
            init_goal_list=[]
            circle_radius = 8.

            groups_pose = []
            groups_goal = []
            for i in self.groups:
                while True:            
                    angle = np.random.random() * np.pi * 2
                    px = circle_radius * np.cos(angle)
                    py = circle_radius * np.sin(angle)
                    gx = -px
                    gy = -py
                    collide = False
                    for grp_pose, grp_goal in zip(groups_pose, groups_goal):
                        #min_dist = 1
                        min_dist = 3  # 220107
                        if np.linalg.norm((px-grp_pose[0],py-grp_pose[1])) < min_dist or np.linalg.norm((gx-grp_goal[0],gy-grp_goal[1]))<min_dist:
                            collide=True
                            break
                        #print(grp_pose[0],grp_pose[1],grp_goal[0],grp_goal[1])
                    if not collide:
                        break
                groups_pose.append([px,py])
                groups_goal.append([gx,gy])
            
            for grp_index, group in enumerate(self.human_list):
                for i in group:                   
                    while True:
                        angle = np.random.random()* np.pi * 2
                        noise = np.random.normal(scale=0.5)
                        #print('group:',index,'i:',i)
                        #print('group pose:',groups_pose)
                        px = groups_pose[grp_index][0]+noise*np.cos(angle)
                        py = groups_pose[grp_index][1]+noise*np.sin(angle)
                        gx=-px
                        gy=-py
                        collide=False
                        
                        for pose, goal in zip(init_pose_list, init_goal_list):
                            min_dist = 1
                            if np.linalg.norm((px-pose[0],py-pose[1])) < min_dist or np.linalg.norm((gx-goal[0],gy-goal[1])) < min_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    theta = np.arctan2(gy, gx)
                    init_pose_list.append([px,py,theta])
                    init_goal_list.append([gx,gy])
                    
        elif rule == 'square_crossing':   #         Rule square_crossing: generate start/goal position at two sides of y-axis
            pass
        elif 'circle_crossing':
            pass

        return init_pose_list, init_goal_list
        
        
    def set_init_pose(self, init_pose):
        self.control_pose(init_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose
        rospy.sleep(0.01)
        
    
    def set_init_goal(self, init_goal):
        self.goal_point = [init_goal[0],init_goal[1]]                 
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)
        
    def point_to_segment_dist(self, x1, y1, x2, y2, x3, y3):
        """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)
    """
        px = x2 - x1
        py = y2 - y1

        if px == 0 and py == 0:
            return np.linalg.norm((x3-x1, y3-y1))

        u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        # (x, y) is the closest point to (x3, y3) on the line segment
        x = x1 + u * px
        y = y1 + u * py

        return np.linalg.norm((x - x3, y-y3))

    def dist(x1, y1, x2, y2):
        return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )