# from GUNHEE
from re import S
import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import rvo2
import pysocialforce as psf
import cv2
import copy

from dbscan.dbscan_new import DBSCAN_new
from dbscan.hdbscan_new import HDBSCAN_new
from dbscan.hdbscan_new_rot import HDBSCAN_new_rot

from numpy import dot
from numpy.linalg import norm

clustering = 'HDBSCAN_new_rot'



# 220725
def generate_pedestrain_list(env, pose_list, velocity_list):
    
    
    pose_list = np.asarray(pose_list)    
    robot_rot = pose_list[0,2]
    pose_list = np.asarray(pose_list[:,0:2]) 
    velocity_list = np.asarray(velocity_list)     # 220105 robot+human poly speed
    
    # pedestrain list 저장공간 생성
    pedestrain_list = np.zeros((pose_list.shape[0], 7))   # 11,7   # px,py,vx,vy,grp,visible
    
    robot_rot += np.pi*3/2   # 220125
    
    visible_ped_pose_list = []
    visible_ped_poly_vel_list = []    
    
    # 로봇 기준 좌표계로 변환(relative pose in cartesian)
    diff_pose = pose_list-pose_list[0]   # 로봇 기준 상대 위치
    #print('pose list:',pose_list)             # debug 로봇~사람들 pose_list x,y
    
    # 로봇 기준 좌표계료 변환(relative velocity in cartesian)
    diff_v = []
    for i, vel in enumerate(velocity_list):  # (magnitude, heading)
        vx_cart_0, vy_cart_0 = pol2cart(velocity_list[0][0],velocity_list[0][1])
        vx_cart, vy_cart = pol2cart(vel[0], vel[1])
        vx_cart -= vx_cart_0
        vy_cart -= vy_cart_0
        diff_v.append([vx_cart, vy_cart])
        
    
    #diff_v = velocity_list - velocity_list[0]   # 로봇 기준 상대속도 magnitude, heading 차이
    
    pose_encoding= np.zeros((pose_list.shape[0]))

    
    labels = []
    for i, (pose,vel) in enumerate(zip(diff_pose, diff_v)):
        diff = pose
        diff_v = vel
        dx_rot = diff[0]*np.cos(robot_rot)+diff[1]*np.sin(robot_rot)
        dy_rot = -diff[0]*np.sin(robot_rot)+diff[1]*np.cos(robot_rot)
        dvx_rot = diff_v[0]*np.cos(robot_rot)+diff_v[1]*np.sin(robot_rot)
        dvy_rot = -diff_v[0]*np.sin(robot_rot)+diff_v[1]*np.cos(robot_rot)
        pedestrain_list[i,0]=dx_rot
        pedestrain_list[i,1]=dy_rot
        pedestrain_list[i,2]=dvx_rot
        pedestrain_list[i,3]=dvy_rot

        
        ped_rot = np.arctan2(dy_rot, dx_rot)
        #print(i,'번째 로테이션:',ped_rot,'의 위치(x,y):',dx_rot, dy_rot, '의 속도(vx,vy):',dvx_rot, dvy_rot)
        
        # 2207250 수정. 1) 로봇은 sensor rotation 범위 내 2) 로봇은 sensor range(6m) 내 3) 사람일 경우(i>0)
        if ped_rot>=0 and ped_rot<=np.pi and np.sqrt(dx_rot**2+dy_rot**2)<=6.0 and i != 0:
            #xx=copy.deepcopy(dx_rot)   # rel.x
            #yy=copy.deepcopy(dy_rot)   # rel.y
            ### relative cartesian velocity를 relative polar velocity로 변환
            #polar_mag, polar_ang = cart2pol(dvx_rot, dvy_rot)
            #vx=copy.deepcopy(polar_mag)
            #vy=copy.deepcopy(polar_ang)
            #vx=copy.deepcopy(vel[0])   # rel.vel.mag
            #vy=copy.deepcopy(vel[1])   # rel.vel.ang
            xx, yy = dx_rot, dy_rot
            vx, vy = dvx_rot, dvy_rot
            visible_ped_pose_list.append([xx, yy])
            visible_ped_poly_vel_list.append([vx,vy])
            pose_encoding[i]=i
            pedestrain_list[i,4]=i # init indiv index
            pedestrain_list[i,5]=i # init grp index
            pedestrain_list[i,6]=1 # visible human
    #print('페드 포드:',visible_ped_pose_list,'페도벨:',visible_ped_poly_vel_list)
    
    
    # 2. Start Unsupervised Clustering
    if visible_ped_pose_list != [] and visible_ped_poly_vel_list != []:
        if clustering=='DBSCAN':
        # original DBSCAN_new  before 220207
            dbscan_new = DBSCAN_new(np.array(visible_ped_pose_list), np.array(visible_ped_poly_vel_list),2,2)
            labels = dbscan_new.grouping(np.array(visible_ped_pose_list), np.array(visible_ped_poly_vel_list))
            #print('라벨:',labels)   # e.g. [0,1,1]
        # after HDBSCAN* 220208?
        elif clustering == 'HDBSCAN':
            hdbscan_new = HDBSCAN_new(np.array(visible_ped_pose_list), np.array(visible_ped_poly_vel_list),2,2)
            
            labels = hdbscan_new.grouping(np.array(visible_ped_pose_list), np.array(visible_ped_poly_vel_list))
            #print('라벨:',labels)   # e.g. [0,1,1]
        # new HDBSCAN_w_rot 220725~
        elif clustering == 'HDBSCAN_new_rot':
            hdbscan_new = HDBSCAN_new_rot(np.array(visible_ped_pose_list), np.array(visible_ped_poly_vel_list))
            
            labels = hdbscan_new.grouping(np.array(visible_ped_pose_list), np.array(visible_ped_poly_vel_list))
            #print('라벨:',labels)   # e.g. [0,1,1]
        else:
            print('clustering error')
        idx = labels
    #print('[DEBUG] 클러스터링 결과 labels:', labels)
    row,axis= np.where(pedestrain_list[:,4:5]!=0.)   # init_grp_index is not 0
    
    for indexx, i in enumerate(row):
        pedestrain_list[i,5]=labels[indexx]+1   # 식별된 그룹 labels [0,0,1] -> [1,1,2] in row_num in pedestrain_list
        
    pedestrain_list = np.array(pedestrain_list)
    
    return pedestrain_list
    # 여기까지가 PPO/generate_action_corl()
    

#https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y