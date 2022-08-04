#### 220228 boundary_dist에서 velocity *= 1.2로 바꿨었음! 기존에는 1.0 (아예 없었음!!) performance에 미치는 영향!!

import time

from sympy import re
import rospy
import copy
import tf
import numpy as np


import lidar_to_grid_map as lg
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import plot, draw, show


from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
import message_filters

from scipy.spatial import ConvexHull





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

        self.robot_value = 10.   # ????
        self.goal_value = 0.     # ????
        # self.reset_pose = None

        self.init_pose = None
        self.flow_map = None  # 211026
        
        #self.time_limit = 750  # before 220210
        self.time_limit = 500   # after 220210
        
        # 0. scanning method
        #self.clustering_method = 'DBSCAN' # 'DBSCAN' befor 220207
        self.clustering_method = 'HDBSCAN' # 'HDBSCAN' after 220207
        
        # 1.Select scenario
        ##Simple scenario
        #self.scenario = 'CC_h5' 
        ## Narrow corrido
        #self.scenario = 'GrpCorridor_h8_grp3'     # CC_h5, GrpCC_h10_grp3, GrpCC_h13_grp4, GrpCC_h21_grp5  ||  GrpCorridor_h8_grp3  || GrpCross_h14_grp4
        ## Group Circle Cross(원)
        self.scenario = 'GrpCC_h10_grp3'     # 써클, 그룹3, 10명
        #self.scenario = 'GrpCC_h13_grp4'     # CC_h5, GrpCC_h10_grp3, GrpCC_h13_grp4, GrpCC_h21_grp5  ||  GrpCorridor_h8_grp3  || GrpCross_h14_grp4
        #self.scenario = 'GrpCC_h15_grp4'     # 0222 for test
        #self.scenario = 'GrpCC_h21_grp5'     # 써클, 그룹 5, 21명
        ## Group Cross Cross(십자)
        #self.scenario = 'GrpCross_h14_grp4'     # CC_h5, GrpCC_h10_grp3, GrpCC_h13_grp4, GrpCC_h21_grp5  ||  GrpCorridor_h8_grp3  || GrpCross_h14_grp4
        # Group Station (역, 아래입구 as main, NW 입구 as 상행, NE 입구 as 하행), Main->NE or NW, NE -> Main, NW -< Main
        #self.scenario = 'GrpStation_h22_grp4'
              
        if self.scenario == 'GrpCorridor_h8_grp3':
            self.rule = 'group_corridor_crossing'
            self.num_human = 9        # Corridor, h8, grp2
            self.groups = [0, 1, 2, 3]              # Corridor, h8, grp2 
            self.human_list = [[0],[1,2,3],[4,5],[6,7,8]]        # Corridor, h8, grp2
            self.time_limit = 500
        
        elif self.scenario == 'GrpCross_h14_grp4':
            self.rule = 'group_cross_crossing'
            self.num_human = 15        # Corridor, h8, grp2
            #self.num_human = 14        # Corridor, h8, grp2
            self.groups = [0, 1, 2, 3, 4]    # CC, h5
            self.human_list = [[0],[1,2,3,4],[5,6,7],[8,9,10,11,12],[13,14]]     # CC, h5
            #self.human_list = [[0],[1,2,3,4],[5,6,7],[8,9,10,11],[12,13]]     # CC, h5
            self.time_limit = 500
            
            
        elif self.scenario == 'CC_h5':
            self.rule = 'group_circle_crossing'
            self.num_human = 6        # Corridor, h8, grp2
            self.groups = [0, 1, 2, 3, 4, 5]    # CC, h5
            self.human_list = [[0],[1],[2],[3],[4],[5]]     # CC, h5
        elif self.scenario == 'GrpCC_h10_grp3':
            self.rule = 'group_circle_crossing'
            self.num_human = 11       # GC, h10, grp3
            self.groups = [0, 1, 2, 3]           # GC, h10, grp3
            self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10]]                  # GC, h10, grp3
        elif self.scenario == 'GrpCC_h13_grp4':
            self.rule = 'group_circle_crossing'
            self.num_human = 14      # GC, h13, grp4          
            self.groups = [0, 1, 2, 3, 4]       # GC, h13, grp4        
            self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10],[11,12,13]]             # GC, h13, grp4       
        elif self.scenario == 'GrpCC_h21_grp5':
            self.rule = 'group_circle_crossing'
            self.num_human = 22      # GC, h21, grp5
            self.groups = [0, 1, 2, 3, 4, 5]    # GC, h21, grp5
            self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10],[11,12,13], [14,15,16,17,18,19,20,21]]   # GC, h21, grp5
        elif self.scenario == 'GrpCC_h15_grp4':
            self.rule = 'group_circle_crossing'
            self.num_human = 16      # GC, h21, grp5
            self.groups = [0, 1, 2, 3, 4]    # GC, h21, grp5
            self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10],[11,12,13,14,15]]   # GC, h21, grp5
            
        elif self.scenario == 'GrpStation_h22_grp4':
            self.rule = 'group_station_crossing'
            self.num_human = 23   
            self.groups = [0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]    # station, h22, group 2+2+2
            #self.human_list=[[0],[1,2,3,4,5],[6,7,8],[9,10,11],[12,13,14,15],[16,17,18],[19,20,21,22]]   # GC, h21, grp5
            self.human_list=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22]]
        else:
            print('Wrong Scenario')
        
        
        # Define Subscriber
        sub_list = []          # https://velog.io/@suasue/Python-%EA%B0%80%EB%B3%80%EC%9D%B8%EC%9E%90args%EC%99%80-%ED%82%A4%EC%9B%8C%EB%93%9C-%EA%B0%80%EB%B3%80%EC%9D%B8%EC%9E%90kwargs
        sub_crash_list = []
        
        for i in range(self.num_human):
        #for i in range(21):   # 220102
            #sub = message_filters.Subscriber('robot_' + str(i) + '/base_pose_ground_truth', Odometry)
            sub = message_filters.Subscriber('robot_' + str(i) + '/odom', Odometry)   # 220720  http://wiki.ros.org/stage_ros
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

        self.is_dead = False

        # -----------Publisher-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)
        
        # ---------Subscriber-----------------        
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
                or self.speed_GT is None or self.state_GT is None or self.speed_poly is None:   # self.speed_poly는 내가 넣은듯?
            #print(self.scan)  # CZwahn이 넣은거
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
            vx=msg.twist.twist.linear.x    # Linear speed
            #vy=msg.twist.twist.linear.y
            #vy=msg.twist.twist.angular.z   # 220725 Angular speed https://www.programcreek.com/python/example/70251/geometry_msgs.msg.Twist
            vy=Euler[2]  # 220725. 각속도를 그냥 heading으로 간주함. 왜냐하면 만약 vel이 polar에서 (1,0)일 경우, cartesian에서 표현 불가 
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
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]   # added
        self.speed_poly = [v_x, v_y]   # added


    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1
        
    def laser_dead_callback(self, dead_scan):   # no meaning?
        self.scan_dead = np.array(dead_scan.ranges)
        self.scan_dead[np.isnan(self.scan_dead)] = 0.6
        self.scan_dead[np.isinf(self.scan_dead)] = 0.6
        dead_scan_min = np.min(self.scan_dead)
        if dead_scan_min < 0.6:
            self.is_dead = True


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
    
    def get_env_pose(self):   # added
        return self.pose_list

    def get_self_speed_poly(self):   # added
        return self.speed_poly

    def get_self_state_rot(self):   # added
        return self.state_GT[2]

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)  # from laser_scan_callback   # self.scan = np.array(scan.ranges)
        scan[np.isnan(scan)] = 6.0       # NaN or INF set 6    # CZwang은 10.0으로 함
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
        scan_sparse = scan_sparse[::-1]    # 211115  fliped input    # 이거 내가 추가했던거 TODO 

        return scan_sparse / 6.0 - 0.5   # because sensor are front of robot(50cm)
    
    # 220711 from ppo.py. generate_action_baseline_LM()
    def get_pedestrain_observation(self, pose_list, velocity_list):
        # Build occupancy map
        cell_size=1*0.5   # Resolution
        map_size=6        # Lidar Sensor Arrange
        local_maps = []
        
        pose_list = np.asarray(pose_list)
        robot_rot = pose_list[0,2]
        robot_rot += np.pi*3/2   # 220125   # from generate_action_corl
    
    
        speed_poly_list = np.asarray(velocity_list)     # 220105 robot+human poly speed
        
        local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))   # [-3~3, 0~6]
        
        #for j in range(3):  # pos, velx,vely
        for j in range(2):  # velx, vely
            for i, pose in enumerate(pose_list):
                diff = pose-pose_list[0]   # 0[0,0], ~, 13[-0.232, -9.2323]
                #print('로봇 rot:',robot_rot)
                # 220110 추가. 로봇 현재 rotation에 따라 변화하는 LM
                dx_rot = diff[0]*np.cos(robot_rot)+diff[1]*np.sin(robot_rot)
                dy_rot = -diff[0]*np.sin(robot_rot)+diff[1]*np.cos(robot_rot)
                
                diff_vel = speed_poly_list - speed_poly_list[0]
                dvx_rot = diff_vel[0]*np.cos(robot_rot)+diff_vel[1]*np.sin(robot_rot)
                dvy_rot = -diff_vel[0]*np.sin(robot_rot)+diff_vel[1]*np.cos(robot_rot)
                
                # 220110 추가. 로봇은 전방 6m만 바라봄(전방x-axis 0~6m, 가로세로y-axis -3~3m)
                # 현재 아래 mod_diff는 로봇 중심 정사각형(360도)로 볼수 있게 한거
                mod_diff_x = np.floor((dx_rot+3)/cell_size)
                #mod_diff_y = np.ceil((map_size/2-dy_rot)/cell_size)
                mod_diff_y = np.ceil((map_size-dy_rot)/cell_size)
                            
                
                #print('index max:',index)
                
                if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0:
                    '''
                    if j==0:   # grp idx    
                        # 220110 추가. pose occpuancy(1) 대신 group occupancy(group id)
                        local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=1  # 220121 그룹별 평균 거리 역순으로 들어감(가까울수록 큰 거리)
                    # 220110 수정. 로봇 rotation에 따라 변환된 vx, vy 들어감
                    elif j==1: # vel x
                        local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot)
                        #print(i,'번째 사람의 vx:',diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot))
                    elif j==2: # vel y
                        local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot)
                        #print(i,'번째 사람의 vy:',-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot))
                    '''
                    if j==0:   # grp idx    
                        local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot)
                    elif j==1: # vel x
                        local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot)
                        #print(i,'번째 사람의 vy:',-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot))
            local_maps.append(local_map.tolist())
            local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))
            
        local_maps = np.array(local_maps) # 3,60,60
        return local_maps

    # ADDED
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

    def get_goal_point(self):  # 211102 ADDED
        return self.goal_point
    
    def get_sensor_map(self):  # ADDED
        return self.map1

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
        rospy.sleep(1.0)   # 211214 이게 원래 디폴트로 되어있던거임

    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()   # generate goal 1) dist to zero > 9, 2) 8<dist to agent<10
        self.goal_point = [x_g, y_g]                 # set "global" goal
        [x, y] = self.get_local_goal()               # calculate local(robot's coord) goal

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)   # dist to local goal
        self.distance = copy.deepcopy(self.pre_distance)

    # TODO: Reward reshape: penalty for circuling around
    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0
        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 5
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -1.
            result = 'Crashed'

        if self.is_dead == 1:
            terminate = True
            reward_c = -1.
            result = 'Crashed'

        reward = reward_g + reward_c 

        if t >= 1000 and (not is_crash or not self.is_dead) :
            reward = -0.1 
            terminate = True
            result = 'Time out'

        return reward, terminate, result
    
    def boundary_dist(self, velocity, rel_ang, laser_flag, const=0.354163):
        # Parameters from Rachel Kirby's thesis
        #front_coeff = 1.0
        front_coeff = 2.0
        side_coeff = 2.0 / 3.0
        rear_coeff = 0.5
        safety_dist = 0.5
        velocity_x = velocity[0]
        velocity_y = velocity[1]

        velocity_magnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        variance_front = max(0.5, front_coeff * velocity_magnitude)
        variance_side = side_coeff * variance_front
        variance_rear = rear_coeff * variance_front

        rel_ang = rel_ang % (2 * np.pi)
        flag = int(np.floor(rel_ang / (np.pi / 2)))
        if flag == 0:
            prev_variance = variance_front
            next_variance = variance_side
        elif flag == 1:
            prev_variance = variance_rear
            next_variance = variance_side
        elif flag == 2:
            prev_variance = variance_rear
            next_variance = variance_side
        else:
            prev_variance = variance_front
            next_variance = variance_side

        dist = np.sqrt(const / ((np.cos(rel_ang) ** 2 / (2 * prev_variance)) + (np.sin(rel_ang) ** 2 / (2 * next_variance))))
        dist = max(safety_dist, dist)

        # Offset pedestrian radius
        if laser_flag:
            dist = dist - 0.5 + 1e-9

        return dist

    
    def draw_social_shapes(self, position, velocity, laser_flag, const=0.35):
        # This function draws social group shapes
        # given the positions and velocities of the pedestrians.

        total_increments = 20 # controls the resolution of the blobs
        #total_increments = 80 # controls the resolution of the blobs  #0228 리포트때 480으로 함
        quater_increments = total_increments / 4
        angle_increment = 2 * np.pi / total_increments

        # Draw a personal space for each pedestrian within the group
        contour_points = []
        for i in range(len(position)):
            center_x = position[i][0]
            center_y = position[i][1]
            velocity_x = velocity[i][0]
            velocity_y = velocity[i][1]
            velocity_angle = np.arctan2(velocity_y, velocity_x)

            # Draw four quater-ovals with the axis determined by front, side and rear "variances"
            # The overall shape contour does not have discontinuities.
            for j in range(total_increments):

                rel_ang = angle_increment * j
                value = self.boundary_dist(velocity[i], rel_ang, laser_flag, const)
                #value *= 1.2  # 0228 리포트때는 1.2배 함
                addition_angle = velocity_angle + rel_ang
                x = center_x + np.cos(addition_angle) * value
                y = center_y + np.sin(addition_angle) * value
                contour_points.append((x, y))
                #print('컨투어 포인트:',j,x,y)

        # Get the convex hull of all the personal spaces
        convex_hull_vertices = []
        hull = ConvexHull(np.array(contour_points))
        for i in hull.vertices:
            hull_vertice = (contour_points[i][0], contour_points[i][1])
            convex_hull_vertices.append(hull_vertice)

        return convex_hull_vertices


    def draw_all_social_spaces(self, gp_labels, positions, velocities, laser_flag, const=None):
        all_vertices = []
        all_labels = np.unique(gp_labels)
        for curr_label in all_labels:
            group_positions = []
            group_velocities = []
            for i, l in enumerate(gp_labels):
                if l == curr_label:
                    group_positions.append(positions[i])
                    group_velocities.append(velocities[i])
            if const == None:
                vertices = self.draw_social_shapes(group_positions, group_velocities, laser_flag)
            else:
                vertices = self.draw_social_shapes(group_positions, group_velocities, laser_flag, 
                                                    const)
            all_vertices.append(vertices)
        return all_vertices       # 0, 1, 2, 3번그룹
    
    
    
    def get_reward_and_terminate_corl(self, t, scaled_action, policy_list, pedestrian_list):   # t is increased 1, but initializezd 1 when terminate=True
        terminate = False
        laser_scan = self.get_laser_observation()   # new laser scan(Because excuted action)
        [x, y, theta] = self.get_self_stateGT()     # "updated" current state
        [v, w] = self.get_self_speedGT()            # updated current velocity
        self.pre_distance = copy.deepcopy(self.distance)   # previous distance to local goal
        
        # Propotional Reward
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)  # updated new distance to local goal after action
        reward_g = (self.pre_distance - self.distance) * 2.5  # REWARD for moving forward, later reach goal reward(+15)  # original
        reward_c = 0  # collision penalty
        reward_w = 0  # too much rotation penalty
        result = 0
        is_crash = self.get_crash_state()   # return self.is_crashed

        if self.distance < self.goal_size:  # success reward
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:                   # collision penalty
            terminate = True
            reward_c = -15.
            result = 'Crashed(ROS)'
        
        
        '''
        # 220119. 관측된 라이다 거리에 반비례해서 penalty linear하게 받게. for 충돌 회피. 1 = 0, 0.8 = 0.2, 0.6 = 0.4
        kkk = self.get_min_lidar_dist()
        penalty_lidar = 0.
        if kkk <= 1.0:
            penalty_lidar = (-1. + kkk)/10
        '''
        
        
        
        min_dist_rrr = 10.0   # 220119
        pose_list_np = np.array(self.pose_list)
        rel_dist_list = pose_list_np[:,0:2]-pose_list_np[0,0:2]
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

        # ROTATION PENALTY        
        if np.abs(w) >  1.05:               # rotation penalty
            reward_w = -0.1 * np.abs(w)

        if t > self.time_limit:  # timeout check  220205 For Group corridor
        #if t > 1000:  # timeout check  220119 after weekly. our가 TO 더 높게 나와서, 더 크게 줌
            terminate = True      
            result = 'Time out'
        
        #print(pedestrian_list[1][6])
        # GRP
        indiv_labels=[]                
        grp_labels=[]
        positions = []
        velocities = []
        rotations = [] # 220723
        
        for idx, ped in enumerate(pedestrian_list):
            if pedestrian_list[idx][6] != 0:
                #print(pedestrian_list[idx])
                indiv_label = int(pedestrian_list[idx][4])
                grp_label = int(pedestrian_list[idx][5])
                position = [pedestrian_list[idx][0],pedestrian_list[idx][1]]
                velocity = [pedestrian_list[idx][2],pedestrian_list[idx][3]]
                indiv_labels.append(indiv_label)
                grp_labels.append(grp_label)
                positions.append(position)
                velocities.append(velocity)
                rotation = [np.arctan2(pedestrian_list[idx][1],pedestrian_list[idx][0])]    # 220723
                rotations.append(rotation)   # 220723

        #print('++++++++++++++++++++++++++++++')
        #print('ind_label:',indiv_labels)    # [6, 8]    [1, 2, 3, 5, 9]
        #print('grp_label:',grp_labels)      # [1, 1]    [1, 3, 1, 1, 2]
        #print('pose:',positions)       # [[1,1],[2,2]]    [[2.8896498104205963, 2.6215839730271973], [0.5328502413870305, 1.8637225827143853], [2.2420605229240325, 3.9390001662482153], [2.7590289592251427, 1.4688141316473313], [1.3278307894609154, 1.2985722552877585]]
        #print('vel:',velocities)      # [[0.2,0.2],[0.2,0.2]]   [[-0.511329087694693, -0.5343655529166533], [-0.6367084601515689, -0.4843576537760276], [-0.3653859186755444, -0.512861404743211], [-0.549097788376088, -0.47435513775756494], [-0.3493583111982035, 0.33289796577453096]]
        #print('rot:',rotations)       # 220723 East(0) ~ West (3.14?)

        ## 220303 HDBSCAN 결과 비쥬얼라이즈 (작동함)
        #img = np.zeros([12,12,3])  # 20 x 20 
        img = np.zeros([24,24,3])  # 20 x 20 
        img[:,:,0]=0
        img[:,:,1]=128
        img[:,:,2]=0
        for i in range(len(indiv_labels)):
            img[23-int(positions[i][1]),int(positions[i][0]+12),0]=grp_labels[i]*25 /255.0
            img[23-int(positions[i][1]),int(positions[i][0]+12),1]=grp_labels[i]*50 /255.0
            img[23-int(positions[i][1]),int(positions[i][0]+12),2]=grp_labels[i]*75 /255.0
        hsv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2HSV)
        hsv=cv2.resize(hsv, dsize=(240,240), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow('Group clutering result',hsv)    # gruop clustering result 보여주는 거 그룹 클러스터링 220420
        #cv2.waitKey(1)


        # 1. Individual Drawing
        indiv_space=self.draw_all_social_spaces(indiv_labels, positions, velocities, False)
        #kkk=self.draw_all_social_spaces(indiv_labels, positions, velocities, False)
        indiv_space=np.array(indiv_space)

        '''
        for i in range(indiv_space.shape[0]):
            indiv_space_vertices = np.array(indiv_space[i])
            plt.plot(indiv_space_vertices[:,0],indiv_space_vertices[:,1])
        '''
      
        # 2. Group Drawing
        grp_space=self.draw_all_social_spaces(grp_labels, positions, velocities, False)
        #kkk=self.draw_all_social_spaces(indiv_labels, positions, velocities, False)
        #print(grp_space)
        grp_space=np.array(grp_space)
        
        # 220723. 그룹별 Convex Hull에서 최소, 최대 angle finding
        #print('grp_space.shape[0]:',grp_space.shape[0], grp_space.shape)        
        heading_list = []
        for i in range(grp_space.shape[0]):
            #print('야:',i, grp_space[i])
            max_heading = -999
            min_heading = 999
            for grp_pp in grp_space[i]:
                heading = np.arctan2(grp_pp[1],grp_pp[0])
                #print(heading)
                if heading < min_heading:
                    min_heading = heading
                if heading > max_heading:
                    max_heading = heading
            #print(i,'의:', min_heading, max_heading)
            heading_list.append([min_heading, max_heading])
        #print(heading_list)
        
        #220723 그룹별 최소, 최대 rot 기준 layer masking
        mask_layer = np.zeros(512)
        resolution = 512 / np.pi
        for min_head, max_head in heading_list:
            #print(min_head, max_head)
            # 데이터 정제화
            if min_head<0:
                min_head = 0
            if max_head>np.pi:
                max_head=np.pi
            mask_layer[np.int(resolution*min_head):np.int(resolution*max_head)] = 1  
            mask_layer = mask_layer[::-1]
        #print('mask layer:',mask_layer)
        
        '''
        plt.title("Social group zones")
        
        #### 0303 SGZ visualize!
        plt.clf()
        #print('grp::',grp_space.shape[0], 'grp_labels::',grp_labels)
        plt.ion()  
        for i in range(grp_space.shape[0]):
            #print(i,'번째의 그룹 라벨색:',grp_labels[i])
            ddddd='black'
            if i+1==0:
                ddddd='blue'
            elif i+1==1:
                ddddd='orange'
            elif i+1==2:
                ddddd='green'
            elif i+1==3:
                ddddd='red'
            else:
                ddddd='black'
            grp_space_vertices = np.array(grp_space[i])
            plt.plot(grp_space_vertices[:,0],grp_space_vertices[:,1], color=ddddd)         
        
        for i in range(len(indiv_labels)):  # 개인 도시
            #print(i,'번째의 indiv_labels:',indiv_labels[i],'그룹 라벨:',grp_labels[i])
            ccccc='black'
            if grp_labels[i]==0:
                ccccc='blue'
                plt.plot(positions[i][0],positions[i][1],'o', color='blue')
            elif grp_labels[i]==1:
                ccccc='orange'
                plt.plot(positions[i][0],positions[i][1],'o', color='orange')
                #print('오렌지 색칠')
            elif grp_labels[i]==2:
                ccccc='green'
                plt.plot(positions[i][0],positions[i][1],'o', color='green')
                #print('그린 색칠')
            elif grp_labels[i]==3:
                ccccc='red'
                plt.plot(positions[i][0],positions[i][1],'o', color='red') 
            else:
                ccccc='black'
                plt.plot(positions[i][0],positions[i][1],'o', color='black')
            
        plt.axis([-7, 7, -1, 7])      # [-6, 6, 0, 6]
        
        #plt.show(block=False)
        '''
        
        #print(indiv_space.shape)      # (num_of_indiv, vertice numbers, (pos))   3, 20, 2
        
        # 개인별 리워드
        reward_ind_list = []
        # [1,2,3,4]    # 개인별로 sum
        for i in range(indiv_space.shape[0]):
            #print('야:',i, indiv_space[i])
            dist = np.linalg.norm(indiv_space[i], axis=1)
            #print(dist, np.min(dist))
            reward_ind_list.append(np.min(dist))
        #print(indiv_labels)
        #print('인디 리워드 리스트:', reward_ind_list)
        
        safe_ind_dist = 1
        reward_ind_sum=0.
        for i in reward_ind_list:
            reward_ind = i-safe_ind_dist
            if reward_ind >=0:
                pass
            else:
                reward_ind_sum += reward_ind
        #print('인디뷰주얼 리워듸',reward_ind_list)
        #print('인디비쥬얼 리워드 섬:', reward_ind_sum)

        # 그룹 리워드
        reward_grp_list = []
        # [1,1,1,2]    # 개인별로 sum
        for i in range(grp_space.shape[0]):
            #print('야:',i, indiv_space[i])
            dist = np.linalg.norm(grp_space[i], axis=1)
            #print(dist, np.min(dist))
            reward_grp_list.append(np.min(dist))
        #print(grp_labels)
        #print('그룹 라벨:',grp_labels,'그룹 리웓 리스트:', reward_grp_list)
        
        reward_grp_sum=0.
        safe_grp_dist = 2.
        grp_coefficient = 0.1
        for i in reward_grp_list:
            reward_grp = i-safe_grp_dist
            if reward_grp >=0:
                pass
            else:
                reward_grp_sum += reward_grp
        #print('그룹 리워드 섬:',reward_grp_sum)


        reward_grp_sum *= grp_coefficient
        # OURS final reward
        reward = reward_g + reward_c + reward_w + reward_grp_sum# + reward_static_time
        #print('tot_R:',reward,'r_g:',reward_g,'r_c:',reward_c,'r_w:',reward_w,'r_grp:',reward_grp_sum)

        #return reward, terminate, result   # float, T or F(base), description
        return reward, terminate, result, mask_layer   # 220723
    
    
    # individual score 220217
    def get_reward_and_terminate_corl_ind(self, t, scaled_action, policy_list, pedestrian_list):   # t is increased 1, but initializezd 1 when terminate=True
        terminate = False
        laser_scan = self.get_laser_observation()   # new laser scan(Because excuted action)
        [x, y, theta] = self.get_self_stateGT()     # "updated" current state
        [v, w] = self.get_self_speedGT()            # updated current velocity
        self.pre_distance = copy.deepcopy(self.distance)   # previous distance to local goal
        
        # Propotional Reward
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)  # updated new distance to local goal after action
        reward_g = (self.pre_distance - self.distance) * 2.5  # REWARD for moving forward, later reach goal reward(+15)  # original
        reward_c = 0  # collision penalty
        reward_w = 0  # too much rotation penalty
        result = 0
        is_crash = self.get_crash_state()   # return self.is_crashed

        if self.distance < self.goal_size:  # success reward
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:                   # collision penalty
            terminate = True
            reward_c = -15.
            result = 'Crashed(ROS)'
        
        
        '''
        # 220119. 관측된 라이다 거리에 반비례해서 penalty linear하게 받게. for 충돌 회피. 1 = 0, 0.8 = 0.2, 0.6 = 0.4
        kkk = self.get_min_lidar_dist()
        penalty_lidar = 0.
        if kkk <= 1.0:
            penalty_lidar = (-1. + kkk)/10
        '''
        
        
        
        min_dist_rrr = 10.0   # 220119
        pose_list_np = np.array(self.pose_list)
        rel_dist_list = pose_list_np[:,0:2]-pose_list_np[0,0:2]
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

        # ROTATION PENALTY        
        if np.abs(w) >  1.05:               # rotation penalty
            reward_w = -0.1 * np.abs(w)

        if t > self.time_limit:  # timeout check  220205 For Group corridor
        #if t > 1000:  # timeout check  220119 after weekly. our가 TO 더 높게 나와서, 더 크게 줌
            terminate = True      
            result = 'Time out'
        
        #print(pedestrian_list[1][6])
        # GRP
        indiv_labels=[]                
        grp_labels=[]
        positions = []
        velocities = []
        
        for idx, ped in enumerate(pedestrian_list):
            if pedestrian_list[idx][6] != 0:
                #print(pedestrian_list[idx])
                indiv_label = int(pedestrian_list[idx][4])
                grp_label = int(pedestrian_list[idx][5])
                position = [pedestrian_list[idx][0],pedestrian_list[idx][1]]
                velocity = [pedestrian_list[idx][2],pedestrian_list[idx][3]]
                indiv_labels.append(indiv_label)
                grp_labels.append(grp_label)
                positions.append(position)
                velocities.append(velocity)

        #print(indiv_labels)    # [6, 8]    [1, 2, 3, 5, 9]
        #print(grp_labels)      # [1, 1]    [1, 3, 1, 1, 2]
        #print(positions)       # [[1,1],[2,2]]    [[2.8896498104205963, 2.6215839730271973], [0.5328502413870305, 1.8637225827143853], [2.2420605229240325, 3.9390001662482153], [2.7590289592251427, 1.4688141316473313], [1.3278307894609154, 1.2985722552877585]]
        #print(velocities)      # [[0.2,0.2],[0.2,0.2]]   [[-0.511329087694693, -0.5343655529166533], [-0.6367084601515689, -0.4843576537760276], [-0.3653859186755444, -0.512861404743211], [-0.549097788376088, -0.47435513775756494], [-0.3493583111982035, 0.33289796577453096]]

        # 1. Individual Drawing
        indiv_space=self.draw_all_social_spaces(indiv_labels, positions, velocities, False)
        #kkk=self.draw_all_social_spaces(indiv_labels, positions, velocities, False)
        indiv_space=np.array(indiv_space)

        # 개인별 리워드
        reward_ind_list = []
        # [1,2,3,4]    # 개인별로 sum
        for i in range(len(indiv_labels)):
            #dist = np.linalg.norm(indiv_space[i], axis=1)
            dist = np.sqrt(positions[i][0]**2+positions[i][1]**2)
            reward_ind_list.append(np.min(dist))
        #print(indiv_labels)
        #print('로봇과 개인간 거리:', reward_ind_list)
        
        safe_ind_dist = 2    # 긃과 동일하게
        reward_ind_sum=0.
        for i in reward_ind_list:
            reward_ind = i-safe_ind_dist
            if reward_ind >=0:
                pass
            else:
                reward_ind_sum += reward_ind
        #print('인디뷰주얼 리워듸',reward_ind_list)
        #print('인디비쥬얼 리워드 섬:', reward_ind_sum)
        
        ind_coefficient = 0.1
        reward_ind_sum *= ind_coefficient
        

        # OURS final reward for individual
        reward = reward_g + reward_c + reward_w + reward_ind_sum
        #print('tot_R:',reward,'r_g:',reward_g,'r_c:',reward_c,'r_w:',reward_w,'r_ind:',reward_ind_sum)

        return reward, terminate, result   # float, T or F(base), description
    
    
    

    def reset_pose(self):
        random_pose = self.generate_random_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9     # this lines are for random start pose
        #rospy.sleep(0.01)
        rospy.sleep(1.0)   # 220708 from CZwan
        self.control_pose(random_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:  # np.bas: absolute, compare # generated random pose with topic pose
            [x_robot, y_robot, theta] = self.get_self_stateGT()    # same
            self.control_pose(random_pose)
        #rospy.sleep(0.01)
        rospy.sleep(1.0)   # 220708 from CZwan
        self.is_dead = False


    def control_vel(self, action):   # real action as array[0.123023, -0.242424]. from
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)
        
    # ADDED
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
        #dis = np.sqrt(x ** 2 + y ** 2)
        #while (dis > 9) and not rospy.is_shutdown():
        #    x = np.random.uniform(-9, 9)
        #    y = np.random.uniform(-9, 9)
        #    dis = np.sqrt(x ** 2 + y ** 2)   
        #theta = np.random.uniform(0, 0.5 * np.pi)
        #return [x, y, theta]     
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
        theta = np.random.uniform(0, 0.5 * np.pi)
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
            
            #np.random.seed(5)  #1  4 5  for cherry piccking 0222
            
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
                    
        elif rule == 'group_corridor_crossing':   #         Rule square_crossing: generate start/goal position at two sides of y-axis
            init_pose_list=[[0,-8,np.pi/2],
                            [-4,8,np.pi*3/2],[-2,8,np.pi*3/2],[-0,8,np.pi*3/2],
                            [2,5,np.pi*3/2],[4,5,np.pi*3/2],
                            [-1,11,np.pi*3/2],[0,11,np.pi*3/2],[1,11,np.pi*3/2]]
            init_goal_list=[[0,8],
                            [-4,-8],[-2,-8],[-0,-8],
                            [2,-8],[4,-8],
                            [-1,-11],[0,-11],[1,-11]]
        elif rule == 'group_cross_crossing':
            init_pose_list=[[0,-9,np.pi/2],
                            [-16,1.4,0],[-15.5,0.7,0],[-15.5,-0.7,0],[-16.0, -1.4, 0],
                            #[-5.0,0,0],[-5.0,-1.0,0],[-4.0,-1.0,0],
                            [-5.0,0,0],[-5.0,-1.2,0],[-4.0,-1.2,0],  # 0228
                            #[6.2,0.7,np.pi],[5.0,0.0,np.pi],[5.7,-0.6,np.pi],[7.0,-1.2,np.pi],[7.0,0.0,np.pi],
                            [6.2,0.7,np.pi],[5.0,0.0,np.pi],[5.7,-0.6,np.pi],[7.0,-1.0,np.pi],[7.0,0.0,np.pi],   # 0228
                            #[6.2,0.7,np.pi],[5.0,0.0,np.pi],[5.7,-0.6,np.pi],[7.0,0.0,np.pi],
                            [13.0,1.6,np.pi],[13.2,1.0,np.pi]]
            init_goal_list=[[0,9],
                            [16,-1.4],[16.0,-0.7],[16.0,0.7],[16.0, 1.4],
                            [16.0,0],[16.0,1.0],[16.0,1.0],
                            [-16.2,-0.7],[-16.0,0.0],[-16.7,0.6],[-16.0,1.2],[-16.5,0.0],
                            #[-16.2,-0.7],[-16.0,0.0],[-16.7,0.6],[-16.5,0.0],
                            [-16.0,-1.6],[-16.2,-1.0]]
        elif rule == 'group_station_crossing':
            init_pose_list=[[8,-6,np.pi/2],
                            [-1,-11.0,np.pi/2],[0.0,-11.0,np.pi/2],[1.0,-11.0,np.pi/2],[-0.5, -11.5, np.pi/2],[0.5, -11.5, np.pi/2],
                            [-1.0,-15.0,np.pi/2],[0.0,-15.0,np.pi/2],[1.0,-15.0,np.pi/2],
                            [-5.0,11.0,np.pi/2*3],[-6.0,11.0,np.pi/2*3],[-5.0,12.0,np.pi/2*3],
                            [-5.0,14.0,np.pi/2*3],[-6.0,14.0,np.pi/2*3],[-5, 15, np.pi/2*3],[-6, 15, np.pi/2*3],
                            [5, 11, np.pi/2*3], [6, 11, np.pi/2*3], [5, 12, np.pi/2*3],
                            [5, 14, np.pi/2*3], [6, 14, np.pi/2*3], [5, 15, np.pi/2*3], [6, 15, np.pi/2*3],
                            ]
            init_goal_list=[[-8,6],
                            [-5,14.0],[-6.0,14.0],[5.0,15.0],[5, 15.0],[-5.0, 16.0], # M -> NW
                            #[8.0,15.0],[9.0,15.0],[8.0,15.0], # M -> NE
                            [7.0,6.0],[6.0,6.0],[8.0,6.0], # M -> NE
                            [-1.0,-15.0],[0.0,-14.0],[1.0,-13.0],  # NW -> M
                            [-2.0,-14.0],[-1.0,-14.0],[1, -14],[2, -14],  # NW -> M
                            [-1.0,-15.0],[0.0,-13.0],[1.0,-15.0],   # NE -> M
                            [-2.0,-14.0],[-1.0,-14.0],[1, -14],[2, -14],  # NE -> M
                            ]
            # 사람 골 재선정 
            #px_list = np.array([-6,6])
            #py_list = np.array([-6,6])
            px = np.random.choice([-6,6], 1)
            py = np.random.choice([-6,6], 1)
            init_pose_list[0]=[px[0],py[0],np.pi/2]
            init_goal_list[0]=[-px[0],-py[0]]
            #init_pose_list[0]=[6,-6,np.pi/2]  # 왜넣었지?
            #init_goal_list[0]=[-6,6]


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