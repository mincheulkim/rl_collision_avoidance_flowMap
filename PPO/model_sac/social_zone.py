

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


################# CORL ############################
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



def create_group_mask_layer(self, pedestrian_list):   # 220725
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
    indiv_space=self.draw_all_social_spaces(grp_labels, positions, velocities, laser_flag = False)
    indiv_space=np.array(indiv_space, dtype=object)

    '''
    for i in range(indiv_space.shape[0]):
        indiv_space_vertices = np.array(indiv_space[i])
        plt.plot(indiv_space_vertices[:,0],indiv_space_vertices[:,1])
    '''
    
    # 2. Group Drawing
    grp_space=self.draw_all_social_spaces(grp_labels, positions, velocities, laser_flag = False)
    #print(grp_space)
    grp_space=np.array(grp_space, dtype=object)
    
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

    return mask_layer   # 220723



