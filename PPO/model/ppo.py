# from GUNHEE
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


hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


def transform_buffer(buff):
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff:
        for state in e[0]:
            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch

def transform_buffer_stacked_LM(buff):   # 211214
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch, local_maps_batch = [], [], [], [], [], [], [], [], []
    #v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff:             # robot_state, a, r_list_new, terminal_list_new, logprob, v, LM_stack
        for state in e[0]:     
            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

        local_maps_batch.append(e[6])  # 211214  # 220105  (2048,8,1,3,60,60)


    
    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    local_maps_batch = np.asarray(local_maps_batch)
    print(local_maps_batch.shape,'e')
    return s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch, local_maps_batch

def transform_buffer_baseline_LM(buff):   # 211214
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch, sensor_map_batch, local_maps_batch = [], [], [], [], [], [], [], [], [], []
    #v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff:             # robot_state, a, r_list_new, terminal_list_new, logprob, v, LM_stack
        for state in e[0]:     
            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])
        sensor_map_batch.append(e[6])

        local_maps_batch.append(e[7])  # 211214  # 220105  (2048,8,1,3,60,60)


    
    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)
    sensor_map_batch=np.asarray(sensor_map_batch)
    local_maps_batch = np.asarray(local_maps_batch)
    print('센서맵shape:',sensor_map_batch.shape,'ped map shape:',local_maps_batch.shape)
    return s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch, sensor_map_batch, local_maps_batch


def generate_action(env, state_list, policy, action_bound, mode=False):
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)


        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()


        v, a, logprob, mean = policy(s_list, goal_list, speed_list)
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()

        mean_v = mean.data.cpu().numpy()
        scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
        if mode==True:
            scaled_action = np.clip(mean_v[0], a_min=action_bound[0], a_max=action_bound[1])

    else:
        v = None
        a = None
        scaled_action = None
        logprob = None
    
    return v, a, logprob, scaled_action

def generate_action_LM(env, state_list, pose_list, velocity_list, policy, action_bound, LM_stack, mode=False):   # 211130
    
    s_list, goal_list, speed_list = [], [], []
    for i in state_list:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])
        
    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)
    pose_list = np.asarray(pose_list)
    speed_poly_list = np.asarray(velocity_list)     # 220105 robot+human poly speed
    
    # Build occupancy map
    cell_size=1*0.1
    map_size=6
    local_maps = []
    
    local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))
    for j in range(3):  # pos, velx,vely
        for i, pose in enumerate(pose_list):
            diff = pose-pose_list[0]
            mod_diff_x = np.floor((diff[0]+map_size/2)/cell_size)
            mod_diff_y = np.ceil((map_size/2-diff[1])/cell_size)
            
            if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0:
                if j==0:   # pose occupancy
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=1
                elif j==1: # vel x
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=speed_poly_list[i][0]-speed_poly_list[0][0]
                elif j==2: # vel y
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=speed_poly_list[i][1]-speed_poly_list[0][1]
        local_maps.append(local_map.tolist())
        local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))

    
    local_maps = np.array(local_maps) # 3,60,60
    left_LM = LM_stack.popleft()
    LM_stack.append(local_maps)   # 8,3,60,60?

    diablos = [LM_stack]
    diablos = np.array(diablos)
    #print(diablos.shape)    # 1,8,3,60,60
    
    np.set_printoptions(threshold=np.inf)
    

    s_list = Variable(torch.from_numpy(s_list)).float().cuda()        # 1,3,512
    goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
    speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
    local_maps_torch = Variable(torch.from_numpy(diablos)).float().cuda()    # (1, 8, 3,60,60)   B, S, C, W, H
    #print(s_list.shape, local_maps_torch.shape)
    
    
    
    v, a, logprob, mean = policy(s_list, goal_list, speed_list, local_maps_torch)
    v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
    mean_v = mean.data.cpu().numpy()
    
    scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
    if mode==True:
        scaled_action = np.clip(mean_v[0], a_min=action_bound[0], a_max=action_bound[1])
    
    return v, a, logprob, scaled_action, local_maps, LM_stack   # local_map = np.ndarray type, shape=(60,60)


def generate_action_stacked_LM(env, state_list, pose_list, velocity_list, policy, action_bound, index, mode=False):   # 211213
    s_list, goal_list, speed_list = [], [], []
    #print('grp index:', index)
    
    for i in state_list:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])            

    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)
    pose_list = np.asarray(pose_list)
    
    # Build occupancy map
    cell_size=1*0.1
    map_size=6
    local_maps = []
    
    local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))
    index_max = max(index)   # in case 3    # index_max = 3, index=[1 1 1 1 1 2 2 2 3 3]
    
    # TODO index_map만큼 돌리고, local_maps=[[],[],[]](3개) 또는 [[],[],[],[]](4개) 형태로 나오게 함
    
    #for j in range(3):   # FIXME: this part and 'number 3' is Daechung.
    for j in list(range(1,index_max+1)):   # index_max = 3 => grp0, grp1, grp2
        #print('j:',j)
        for i, pose in enumerate(pose_list):
            diff = pose-pose_list[0]
            mod_diff_x = np.floor((diff[0]+map_size/2)/cell_size)
            mod_diff_y = np.ceil((map_size/2-diff[1])/cell_size)
            
            
            #if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0:
            if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0 and index[i-1]==j:
                #print(i, mod_diff_x, 'original diff:',diff[1], (diff[1]-map_size/2)/cell_size, np.ceil((diff[1]-map_size/2)/cell_size),'abs:',mod_diff_y)
                #if (j==0 and i in [1, 2, 3, 4, 5]) or (j==1 and i in [6, 7, 8]) or (j==2 and i in [9, 10]):  # FIXME. assign humans to groups
                
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=1
            #print('i:',mod_diff_y, mod_diff_x, i)
        local_maps.append(local_map.tolist())
        local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))
    #print(local_map.tolist())   # 211213. [[60],[60],...,[60]]  # https://appia.tistory.com/175
    
    local_maps = [local_maps]   # 211214 to match shape of robot_state (3072, 1, 3, 512) and local_map(30712, 1, 3, 60, 60)
    local_maps = np.array(local_maps)
    
    # 211230 fit max channel size
    #print(local_maps.shape, local_maps.shape[1], index, index_max)
    fit_channel_num = 10
    source = np.zeros((1, 1, 60, 60))
    if local_maps.shape[1] != fit_channel_num:
        for i in range(fit_channel_num-local_maps.shape[1]):
            local_maps = np.append(local_maps, source, axis=1)
            #print('run',-1-i)
    #print('after:',local_maps.shape)
    #print('index:',index, index_max)

    s_list = Variable(torch.from_numpy(s_list)).float().cuda()
    goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
    speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
    local_maps_torch = Variable(torch.from_numpy(local_maps)).float().cuda()

    
    #print(s_list.shape, local_maps_torch.shape) # (1, 3, 512), (1, 3, 60, 60)  # 211213
    '''
    test = local_maps_torch.cpu().numpy()
    print(test[0][0])
    cv2.imshow('map11',cv2.resize(local_maps[0][0], dsize=(480,480), interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(1)
    '''
    
    v, a, logprob, mean = policy(s_list, goal_list, speed_list, local_maps_torch)    # from Stacked_LM_Policy
    v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()

    mean_v = mean.data.cpu().numpy()
        
    scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
    
    if mode==True:
        scaled_action = np.clip(mean_v[0], a_min=action_bound[0], a_max=action_bound[1])    
        
    return v, a, logprob, scaled_action, local_maps   # local_map = np.ndarray type, shape=(60,60)


def generate_action_concat_LM(env, state_list, pose_list, velocity_list, policy, action_bound, LM_stack, index, mode=False):   # 211130
    # t-7 ~ t까지의 8개 time sequence 정보를 받음     LM_stack
    # 각 sequence별 3개 채널: Grop(0), vx(1), vy(2)  index
    s_list, goal_list, speed_list = [], [], []
    for i in state_list:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])
        
    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)
    robot_rot = pose_list[0,2]
    pose_list = np.asarray(pose_list[:,0:2])    # 13+1개
    
    speed_poly_list = np.asarray(velocity_list)     # 220105 robot+human poly speed
    

    # 220121 그룹별 평균 거리 계산    
    rel_dist = pose_list - pose_list[0]
    rel_dist = rel_dist[1:]
    num_dist = np.linalg.norm(rel_dist, axis=1)

    num_grp_dist_array = np.zeros(shape=(np.max(index)+1), dtype=np.float32)
    #print(num_grp_dist_array)
    for idx, grp_dist in zip(index, num_dist):   # 0~13 humans
        #print(idx, grp_dist)
        num_grp_dist_array[idx] += grp_dist
    #print('final:',num_grp_dist_array)
    
    num_cnt_dist_array = np.zeros(shape=(np.max(index)+1), dtype=np.float32)
    for i in range(np.max(index)+1):
        num_cnt_dist_array[i] += index.count(i)
    #print(num_cnt_dist_array)
    num_grp_dist_array = num_grp_dist_array / num_cnt_dist_array
    #print('average:',num_grp_dist_array)
        

    
    
    
    # Build occupancy map
    cell_size=1*0.1
    map_size=6
    local_maps = []
    
    #print('인덱스:',index)    # 0~12: human1~human13  (13)
    
    local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))   # [-3~3, 0~6]
    for j in range(3):  # grp, vel_x,vel_y
        for i, pose in enumerate(pose_list):
            diff = pose-pose_list[0]   # 0[0,0], ~, 13[-0.232, -9.2323]
            # 220110 추가. 로봇 현재 rotation에 따라 변화하는 LM
            dx_rot = diff[0]*np.cos(robot_rot)+diff[1]*np.sin(robot_rot)
            dy_rot = -diff[0]*np.sin(robot_rot)+diff[1]*np.cos(robot_rot)
            
            # 220110 추가. 로봇은 전방 6m만 바라봄(전방x-axis 0~6m, 가로세로y-axis -3~3m)
            mod_diff_x = np.floor((dx_rot)/cell_size)
            mod_diff_y = np.ceil((map_size/2-dy_rot)/cell_size)
                        
            diff_vel = speed_poly_list - speed_poly_list[0]
            #print('index max:',index)
            
            if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0:
                if j==0:   # grp idx    
                    # 220110 추가. pose occpuancy(1) 대신 group occupancy(group id)
                    #local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=(index[i-1]+1)/(np.max(index)+1)  # 220119 grp index starts from 0...
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=1/num_grp_dist_array[index[i-1]]  # 220121 그룹별 평균 거리 역순으로 들어감(가까울수록 큰 거리)
                    #print(index)
                    #print(i,'번째 사람의 index:',index[i-1]+1 , num_grp_dist_array[index[i-1]])
                    #print(i, 1/num_grp_dist_array[index[i-1]])
                # 220110 수정. 로봇 rotation에 따라 변환된 vx, vy 들어감
                elif j==1: # vel x
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot)
                    #print(i,'번째 사람의 vx:',diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot))
                elif j==2: # vel y
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot)
                    #print(i,'번째 사람의 vy:',-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot))
        local_maps.append(local_map.tolist())
        local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))

    
    local_maps = np.array(local_maps) # 3,60,60
    left_LM = LM_stack.popleft()
    LM_stack.append(local_maps)   # 8,3,60,60
    
    diablos = [LM_stack]
    diablos = np.array(diablos)

    np.set_printoptions(threshold=np.inf)
    
    s_list = Variable(torch.from_numpy(s_list)).float().cuda()        # 1,3,512
    goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
    speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
    local_maps_torch = Variable(torch.from_numpy(diablos)).float().cuda()    # (1, 8, 3,60,60)   B, S, C, W, H

    v, a, logprob, mean = policy(s_list, goal_list, speed_list, local_maps_torch)
    v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
    mean_v = mean.data.cpu().numpy()
    
    scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
    if mode==True:
        scaled_action = np.clip(mean_v[0], a_min=action_bound[0], a_max=action_bound[1])
        
    return v, a, logprob, scaled_action, local_maps, LM_stack   # local_map = np.ndarray type, shape=(60,60)



def generate_action_depth_LM(env, state_list, pose_list, velocity_list, policy, action_bound, LM_stack, index, mode=False):   # 211130
    # t-7 ~ t까지의 8개 time sequence 정보를 받음     LM_stack
    # 각 sequence별 3개 채널: Grop(0), vx(1), vy(2)  index
    s_list, goal_list, speed_list = [], [], []
    for i in state_list:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])

    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)
    robot_rot = pose_list[0,2]
    pose_list = np.asarray(pose_list[:,0:2])    # 13+1개
    
    speed_poly_list = np.asarray(velocity_list)     # 220105 robot+human poly speed
    

    # 220121 그룹별 평균 거리 계산    
    rel_dist = pose_list - pose_list[0]
    rel_dist = rel_dist[1:]
    num_dist = np.linalg.norm(rel_dist, axis=1)

    num_grp_dist_array = np.zeros(shape=(np.max(index)+1), dtype=np.float32)
    for idx, grp_dist in zip(index, num_dist):   # 0~13 humans
        num_grp_dist_array[idx] += grp_dist
    
    num_cnt_dist_array = np.zeros(shape=(np.max(index)+1), dtype=np.float32)
    for i in range(np.max(index)+1):
        num_cnt_dist_array[i] += index.count(i)
    num_grp_dist_array = num_grp_dist_array / num_cnt_dist_array
    
    # Build occupancy map
    cell_size=1*0.1
    map_size=6
    local_maps = []

    local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))   # [-3~3, 0~6]
    for j in range(3):  # grp, vel_x,vel_y
        for i, pose in enumerate(pose_list):
            diff = pose-pose_list[0]   # 0[0,0], ~, 13[-0.232, -9.2323]
            # 220110 추가. 로봇 현재 rotation에 따라 변화하는 LM
            dx_rot = diff[0]*np.cos(robot_rot)+diff[1]*np.sin(robot_rot)
            dy_rot = -diff[0]*np.sin(robot_rot)+diff[1]*np.cos(robot_rot)
            
            # 220110 추가. 로봇은 전방 6m만 바라봄(전방x-axis 0~6m, 가로세로y-axis -3~3m)
            mod_diff_x = np.floor((dx_rot)/cell_size)
            mod_diff_y = np.ceil((map_size/2-dy_rot)/cell_size)
                        
            diff_vel = speed_poly_list - speed_poly_list[0]
            #print('index max:',index)
            
            if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0:
                if j==0:   # grp idx    
                    # 220110 추가. pose occpuancy(1) 대신 group occupancy(group id)
                    #local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=(index[i-1]+1)/(np.max(index)+1)  # 220119 grp index starts from 0...
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=1/num_grp_dist_array[index[i-1]]  # 220121 그룹별 평균 거리 역순으로 들어감(가까울수록 큰 거리)
                    #print(index)
                    #print(i,'번째 사람의 index:',index[i-1]+1 , num_grp_dist_array[index[i-1]])
                    #print(i, 1/num_grp_dist_array[index[i-1]])
                # 220110 수정. 로봇 rotation에 따라 변환된 vx, vy 들어감
                elif j==1: # vel x
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot)
                    #print(i,'번째 사람의 vx:',diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot))
                elif j==2: # vel y
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot)
                    #print(i,'번째 사람의 vy:',-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot))
        local_maps.append(local_map.tolist())
        local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))

    
    local_maps = np.array(local_maps) # 3,60,60
    left_LM = LM_stack.popleft()
    LM_stack.append(local_maps)   # 8,3,60,60
    
    diablos = [LM_stack]
    diablos = np.array(diablos)

    np.set_printoptions(threshold=np.inf)
    
    s_list = Variable(torch.from_numpy(s_list)).float().cuda()        # 1,3,512
    goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
    speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
    local_maps_torch = Variable(torch.from_numpy(diablos)).float().cuda()    # (1, 8, 3,60,60)   B, S, C, W, H
    v, a, logprob, mean = policy(s_list, goal_list, speed_list, local_maps_torch)
    v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
    mean_v = mean.data.cpu().numpy()
    
    scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
    if mode==True:
        scaled_action = np.clip(mean_v[0], a_min=action_bound[0], a_max=action_bound[1])
        
    return v, a, logprob, scaled_action, local_maps, LM_stack   # local_map = np.ndarray type, shape=(60,60)


# 220124 IROS baseline
def generate_action_baseline_LM(env, state_list, pose_list, velocity_list, policy, action_bound, LM_stack, index, mode=False):   # 211130
    s_list, goal_list, speed_list = [], [], []
    for i in state_list:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])

    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)
    robot_rot = pose_list[0,2]
    pose_list = np.asarray(pose_list[:,0:2])    # 13+1개
    
    speed_poly_list = np.asarray(velocity_list)     # 220105 robot+human poly speed
    

    # Build occupancy map
    cell_size=1*0.1
    map_size=6
    local_maps = []
    
    robot_rot += np.pi*3/2   # 220125

    local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))   # [-3~3, 0~6]
    for j in range(3):  # grp, vel_x,vel_y
        for i, pose in enumerate(pose_list):
            diff = pose-pose_list[0]   # 0[0,0], ~, 13[-0.232, -9.2323]
            # 220110 추가. 로봇 현재 rotation에 따라 변화하는 LM
            dx_rot = diff[0]*np.cos(robot_rot)+diff[1]*np.sin(robot_rot)
            dy_rot = -diff[0]*np.sin(robot_rot)+diff[1]*np.cos(robot_rot)
            #print('posei:',i,diff,(dx_rot,dy_rot))
            
            # 220110 추가. 로봇은 전방 6m만 바라봄(전방x-axis 0~6m, 가로세로y-axis -3~3m)
            mod_diff_x = np.floor((dx_rot+3)/cell_size)
            mod_diff_y = np.ceil((map_size/2-dy_rot)/cell_size)
            #print('modx,y:',mod_diff_x,mod_diff_y)
                        
            diff_vel = speed_poly_list - speed_poly_list[0]
            #print('index max:',index)
            
            if mod_diff_x >=0 and mod_diff_x <(map_size/cell_size) and mod_diff_y >=0 and mod_diff_y <(map_size/cell_size) and i != 0:
                if j==0:   # grp idx    
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=1
                    #print(dx_rot,dy_rot,'modified x,y:',mod_diff_x,mod_diff_y)
                elif j==1: # vel x
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot)
                    #print(i,dx_rot,dy_rot,'vx:',diff_vel[i][0]*np.cos(robot_rot)+diff_vel[i][1]*np.sin(robot_rot))
                elif j==2: # vel y
                    local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot)
                    #print(i,dx_rot,dy_rot,'vy',-diff_vel[i][0]*np.sin(robot_rot)+diff_vel[i][1]*np.cos(robot_rot))
        local_maps.append(local_map.tolist())
        local_map = np.zeros((int(map_size/cell_size),int(map_size/cell_size)))

    local_maps = [local_maps]
    local_maps = np.array(local_maps) # 3,60,60
    local_maps[:,30:60,:]=0  # masking back area of robot
    
    sensor_map = env.get_sensor_map()   # got sensor map
    sensor_map = [sensor_map]
    sensor_map = [sensor_map]
    sensor_map = np.array(sensor_map)   # (1,60,60)


    np.set_printoptions(threshold=np.inf)
    
    s_list = Variable(torch.from_numpy(s_list)).float().cuda()        # 1,3,512
    goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
    speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
    sensor_map_torch = Variable(torch.from_numpy(sensor_map)).float().cuda()    # 1, 60, 60)
    local_maps_torch = Variable(torch.from_numpy(local_maps)).float().cuda()    # (3,60,60)   B, C, W, H
    
    v, a, logprob, mean = policy(s_list, goal_list, speed_list, sensor_map_torch, local_maps_torch)
    v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
    mean_v = mean.data.cpu().numpy()
    
    scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
    if mode==True:
        scaled_action = np.clip(mean_v[0], a_min=action_bound[0], a_max=action_bound[1])
    
    
    
    '''
    # Visualize sensor and ped maps
    hsv=cv2.resize(sensor_map[0], dsize=(480,480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('sensormap',hsv)
    cv2.waitKey(1)

    hsv=cv2.resize(local_maps[0], dsize=(480,480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('pos',hsv)
    cv2.waitKey(1)
    
    hsv=cv2.resize(local_maps[1], dsize=(480,480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('vx',hsv)
    cv2.waitKey(1)
    
    hsv=cv2.resize(local_maps[2], dsize=(480,480), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('vy',hsv)
    cv2.waitKey(1)
    '''

    return v, a, logprob, scaled_action, sensor_map, local_maps   # local_map = np.ndarray type, shape=(60,60)


def generate_action_no_sampling(env, state_list, policy, action_bound):
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

        _, _, _, mean = policy(s_list, goal_list, speed_list)
        mean = mean.data.cpu().numpy()
        scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
    else:
        mean = None
        scaled_action = None

    return mean, scaled_action

def generate_action_human(env, pose_list, goal_global_list, num_env):   # pose_list added
    if env.index == 0:
        # 211020 pose_list create        
        p_list = []
        for i in pose_list:
            p_list.append(i)
        p_list = np.asarray(p_list)
        
        # Get action for humans(RVO)
        #sim = rvo2.PyRVOSimulator(1/60., num_env, 5, 3, 3, 0.4, 1)
        
        sim = rvo2.PyRVOSimulator(1/60., 3, 5, 5, 5, 0.5, 1)  # 211108   # neighborDist, maxNeighbors, timeHorizon, TimeHorizonObst, radius, maxspeed

        for i in range(num_env):  # i=0, 1,2,3,4
            if i >= 1:
                sim.addAgent(tuple(p_list[i]))
                hv = goal_global_list[i] - p_list[i] 
                hs = np.linalg.norm(hv)     # 211027   get raw_scaled action from learned policy
                prefv=hv/hs if hs >1 else hv
                sim.setAgentPrefVelocity(i-1, tuple(prefv))
                
        # Obstacles are also supported # 211022   https://gamma.cs.unc.edu/RVO2/documentation/2.0/class_r_v_o_1_1_r_v_o_simulator.html#a0f4a896c78fc09240083faf2962a69f2
        #o1 = sim.addObstacle([(2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)])
        #sim.processObstacles()
        # TODO concern about local obstacle

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        sim.doStep()

        scaled_action = []       
        for i in range(num_env):  # num_env=3,  i=0, 1,2
            if i==0:
                scaled_action.append((0,0))
            if i >= 1:
                scaled_action.append(sim.getAgentVelocity(i-1))
        
    else:  # env.index =! 0
        scaled_action = None

    return scaled_action

# 211129
def generate_action_human_groups(env, pose_list, goal_global_list, num_env):
    if env.index == 0:
        # 211020 pose_list create        
        p_list = []
        for i in pose_list:
            p_list.append(i)
        p_list = np.asarray(p_list)
        
        # Get action for humans(RVO)
        #sim = rvo2.PyRVOSimulator(1/60., num_env, 5, 3, 3, 0.4, 1)
        #sim = rvo2.PyRVOSimulator(1/60., 3, 5, 5, 5, 0.5, 1)  # 211108   # neighborDist, maxNeighbors, timeHorizon, TimeHorizonObst, radius, maxspeed
        sim = rvo2.PyRVOSimulator(1/60., 6, 6, 5, 5, 0.5, 1)  # 211108   # neighborDist, maxNeighbors, timeHorizon, TimeHorizonObst, radius, maxspeed


        # 211129 add cohesion force
        tendency = 1
        togo_diff_prefv = [0,0]
        for i in range(num_env):  # i=0, 1,2,3,4
            if i >= 1:
                sim.addAgent(tuple(p_list[i]))   # create total 4 agent(1,2,3,4 -> 0,1,2,3)
                # 1. TOGO force is
                hv = goal_global_list[i] - p_list[i]
                '''
                # 2. Cohesion force
                if i in [2,3]:   # 1 with 2, 3
                    togo_diff = pose_list[1] - pose_list[i]
                    togo_diff_norm = np.linalg.norm(togo_diff)
                    togo_diff_prefv=togo_diff/togo_diff_norm if togo_diff_norm >1 else togo_diff
                elif i in [5]:   # 4 with 5
                    togo_diff = pose_list[4] - pose_list[i]
                    togo_diff_norm = np.linalg.norm(togo_diff)
                    togo_diff_prefv=togo_diff/togo_diff_norm if togo_diff_norm >1 else togo_diff
                '''
                hs = np.linalg.norm(hv)     # 211027   get raw_scaled action from learned policy
                prefv=hv/hs if hs >1 else hv
                '''
                # 3. incorporate toGoal + toGroupCenter
                #print('i:',i,'hv:',)
                print(i,'s prefv:',prefv, togo_diff_prefv)
                prefv = prefv + (togo_diff_prefv*tendency)
                
                prefv_new = np.linalg.norm(prefv)
                prefv = prefv/prefv_new if prefv_new > 1 else prefv
                print(i,'s after prefv:',prefv, togo_diff_prefv)
                '''
                sim.setAgentPrefVelocity(i-1, tuple(prefv))
                
        # Obstacles are also supported # 211022   https://gamma.cs.unc.edu/RVO2/documentation/2.0/class_r_v_o_1_1_r_v_o_simulator.html#a0f4a896c78fc09240083faf2962a69f2
        #o1 = sim.addObstacle([(2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)])
        #sim.processObstacles()
        # TODO concern about local obstacle

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        sim.doStep()

        scaled_action = []       
        for i in range(num_env):  # num_env=3,  i=0, 1,2
            if i==0:
                scaled_action.append((0,0))
            if i >= 1:
                scaled_action.append(sim.getAgentVelocity(i-1))   # rank 6 -> dummy, 0, 1, 2, 3, 4

        
        # 211129 add cohesion force
        tendency = 0.7
        for i, pose in enumerate(scaled_action):   # https://wayhome25.github.io/python/2017/02/24/py-07-for-loop/
            #print('i:',i, 'pose: ',pose)
            if i in [2,3]:   # 1 with 2, 3
                togo_diff = pose_list[1] - pose_list[i]
                togo_diff_norm = np.linalg.norm(togo_diff)     # 211027   get raw_scaled action from learned policy
                togo_diff_prefv=togo_diff/togo_diff_norm if togo_diff_norm >1 else togo_diff
                scaled_action[i] += togo_diff_prefv*tendency
            
            elif i in [5]:   # 4 with 5
                togo_diff = pose_list[4] - pose_list[i]
                togo_diff_norm = np.linalg.norm(togo_diff)     # 211027   get raw_scaled action from learned policy
                togo_diff_prefv=togo_diff/togo_diff_norm if togo_diff_norm >1 else togo_diff
                scaled_action[i] += togo_diff_prefv*tendency
    else:  # env.index =! 0
        scaled_action = None
    
    return scaled_action

def generate_action_human_sf(env, pose_list, goal_global_list, num_env, robot_visible, grp_list):   # 211221    
    human_max_speed = 1.0
    p_list = []
    scaled_action = []
    scaled_position = []

    if robot_visible:    
        for i in pose_list:
            p_list.append(i)
        p_list = np.asarray(p_list)

        # 1. initial states
        initial_state = np.zeros((num_env, 6))
        #print(num_env, 'goal:',goal_global_list, 'plist;',p_list)
        for i in range(num_env):  # i=0, 1,2,3,4        
            hv = goal_global_list[i] - p_list[i] 
            hs = np.linalg.norm(hv)     # 211027   get raw_scaled action from learned policy
            prefv=hv/hs if hs >human_max_speed else hv
            initial_state[i, :]=np.array([p_list[i][0],p_list[i][1], prefv[0], prefv[1], goal_global_list[i][0],goal_global_list[i][1]])
        # ROLLBACK
        # 2. group #################################
        groups = grp_list          # 220118

        # 3. assign obstacles
        #obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
        psf_sim = None
        
        # 4. initiate simulator
        psf_sim = psf.Simulator(
                initial_state, groups=groups, obstacles=None, config_file="./pysocialforce/config/example.toml"
                # TOML doesn't work. modify directly pysocialforce/scene.py
            )
        # do 1 updates
        psf_sim.step(n=1)
        ped_states, group_states = psf_sim.get_states()    # sx, sy, vx, vy, gx, gy, tau

        # 5. visualize
        #with psf.plot.SceneVisualizer(psf_sim, "output_image_sf") as sv:
        #    sv.animate()
        
        for i in range(num_env):
            #print(i,':',ped_states[1][i][0],ped_states[1][i][1],'vel:',ped_states[1][i][2],ped_states[1][i][3])
            vx = ped_states[1][i][2]
            vy = ped_states[1][i][3]
            vx = vx/1
            vy=vy/1
            scaled_action.append([vx,vy])
            scaled_position.append([ped_states[1][i][0],ped_states[1][i][1],0])
                
        return scaled_action, scaled_position
    
    else:     # 220119 robot is not visible
        for i in pose_list:
            p_list.append(i)
        p_list = np.asarray(p_list)

        initial_state = np.zeros((num_env, 6))

        for i in range(num_env):  # i=0, 1,2,3,4        
            hv = goal_global_list[i] - p_list[i] 
            hs = np.linalg.norm(hv)     # 211027   get raw_scaled action from learned policy
            prefv=hv/hs if hs >human_max_speed else hv
            initial_state[i, :]=np.array([p_list[i][0],p_list[i][1], prefv[0], prefv[1], goal_global_list[i][0],goal_global_list[i][1]])
        
        groupss = copy.deepcopy(grp_list)          # 220119

        psf_sim = None
        groups_ex_human = []
        for group_d in groupss[1:]:
            for i in range(len(group_d)):
                group_d[i]=group_d[i]-1
            groups_ex_human.append(group_d)
        
        psf_sim = psf.Simulator(
                initial_state[1:], groups=groups_ex_human, obstacles=None, config_file="./pysocialforce/config/example.toml"
            )
        # do 1 updates
        psf_sim.step(n=1)
        ped_states, group_states = psf_sim.get_states()    # sx, sy, vx, vy, gx, gy, tau

        for i in range(num_env-1):
            vx = ped_states[1][i][2]
            vy = ped_states[1][i][3]
            scaled_action.append([vx,vy])
            scaled_position.append([ped_states[1][i][0],ped_states[1][i][1],0])
        scaled_action.insert(0,[0.0,0.0])     # 220119 for robot
        return scaled_action, scaled_position    


def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]     # 3000
    num_env = rewards.shape[1]      # 1
    
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))    # (3001,1)

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs



def ppo_update_stage1(policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):     # num_step = 1000, batch_size=1024
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()
    print('ppo_update_stage1():',obss.shape, goals.shape, speeds.shape, actions.shape, logprobs.shape, targets.shape, values.shape, rewards.shape, advs.shape)
    obss = obss.reshape((num_step*num_env, frames, obs_size))    # 1000 samples   # reshape as num_step(3000,horizon)*num_env(1), 3, 512
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,   # from 0~999, pick 1024 nums
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            #loss = policy_loss + 0.5 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy, optimizer.param_groups[0]['lr']))

def ppo_update_stage1_stacked_LM(policy, optimizer, batch_size, memory, epoch,    # 211214
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=1, frames=3, obs_size=512, act_size=2):     # num_step = 1000, batch_size=1024
    #obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs, local_mapss = memory

    advs = (advs - advs.mean()) / advs.std()
    print('ppo_update_stage1_stacked_LM():',obss.shape, goals.shape, speeds.shape, actions.shape, logprobs.shape, targets.shape, values.shape, rewards.shape, advs.shape, 'local mapss.shape:',local_mapss.shape)

    obss = obss.reshape((num_step*num_env, frames, obs_size))    # (3072, 1, 3, 512) -> reshape as 3072(num_step=HORIZON) * 1(num_env), 3, 512
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    local_map_width = 60   # 211214
    fix_group_num = 10   # 211230 5 # 220104 10
    local_mapss = local_mapss.reshape((num_step*num_env, fix_group_num, local_map_width, local_map_width))    # -> (2048, 10, 60, 60)
    #print('reshape:',local_mapss.shape)     

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,   # from 0~999, pick 1024 nums
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()  

            sampled_local_maps = Variable(torch.from_numpy(local_mapss[index])).float().cuda()  # 211214

                                                                        # (1024, 3, 512)  (1024, 2)      (1024, 2)        (1024, 2)    be (1024, 3, 60, 60)
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions, sampled_local_maps)   # FIXME

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            #loss = policy_loss + 0.5 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy, optimizer.param_groups[0]['lr']))

    #print('update')
    
def ppo_update_stage1_LM(policy, optimizer, batch_size, memory, epoch,    # 211214
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=1, frames=3, obs_size=512, act_size=2):     # num_step = 1000, batch_size=1024
    #obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs, local_mapss = memory

    advs = (advs - advs.mean()) / advs.std()
    print('ppo_update_stage1_LM():',obss.shape, goals.shape, speeds.shape, actions.shape, logprobs.shape, targets.shape, values.shape, rewards.shape, advs.shape, 'local mapss.shape:',local_mapss.shape)
                                                                                        # in (2048, 1, 8, 3, 60, 60)   -> (2048, 8, 3, 60, 60)

    obss = obss.reshape((num_step*num_env, frames, obs_size))    # (3072, 1, 3, 512) -> reshape as 3072(num_step=HORIZON) * 1(num_env), 3, 512
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    local_map_width = 60   # 211214
    fix_group_num = 3   # 211230 5 # 220104 10
    fix_num_channel = 3 # pos, velx, vely
    fix_seq = 8 # t-7, t-6, ..., t
    local_mapss = local_mapss.reshape((num_step*num_env, fix_seq, fix_num_channel, local_map_width, local_map_width))    # in (2048, 1, 8, 3, 60, 60)  -> TODO(2048, 8, 3, 60, 60)  B S C W H
    

    for update in range(epoch):   # 0, 1
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,   # from 0~999, pick 1024 nums
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()  

            sampled_local_maps = Variable(torch.from_numpy(local_mapss[index])).float().cuda()  # 211214

                                                                        # (1024, 3, 512)  (1024, 2)      (1024, 2)        (1024, 2)    be (1024, 3, 60, 60)   (1024, 8, 3, 60, 60)
            #TOBO new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions, sampled_local_maps)   # TODO
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions, sampled_local_maps)   # Before
            
            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            #loss = policy_loss + 0.5 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy, optimizer.param_groups[0]['lr']))
    

def ppo_update_stage1_baseline_LM(policy, optimizer, batch_size, memory, epoch,    # 211214
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=1, frames=3, obs_size=512, act_size=2):     # num_step = 1000, batch_size=1024
    #obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs, sensor_maps, local_mapss = memory

    advs = (advs - advs.mean()) / advs.std()
    print('ppo_update_stage1_baseline_LM():',obss.shape, goals.shape, speeds.shape, actions.shape, logprobs.shape, targets.shape, values.shape, rewards.shape, advs.shape, 'sensormaps:',sensor_maps.shape, 'local mapss.shape:',local_mapss.shape)
                                                                                        # in (2048, 1, 8, 3, 60, 60)   -> (2048, 8, 3, 60, 60)

    obss = obss.reshape((num_step*num_env, frames, obs_size))    # (3072, 1, 3, 512) -> reshape as 3072(num_step=HORIZON) * 1(num_env), 3, 512
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    local_map_width = 60   # 211214
    fix_num_channel = 3 # pos, velx, vely       # pos, vx, vy for ped map
    
    sensor_maps = sensor_maps.reshape(num_step*num_env, 1, local_map_width, local_map_width)                    # 1024, 1, 60, 60 sensor map
    local_mapss = local_mapss.reshape((num_step*num_env, fix_num_channel, local_map_width, local_map_width))    # in 1024, 3, 60, 60 ped map
    

    for update in range(epoch):   # 0, 1
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,   # from 0~999, pick 1024 nums
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()  
            
            sampled_sensor_maps = Variable(torch.from_numpy(sensor_maps[index])).float().cuda()  # 220124
            sampled_local_maps = Variable(torch.from_numpy(local_mapss[index])).float().cuda()  # 211214

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions, sampled_sensor_maps, sampled_local_maps)   # Before
            
            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            #loss = policy_loss + 0.5 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy, optimizer.param_groups[0]['lr']))


def ppo_update_stage2(policy, optimizer, batch_size, memory, filter_index, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    obss = obss.reshape((num_step*num_env, frames, obs_size))
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    #----------------------------------------------------------
    obss = np.delete(obss, filter_index, 0)
    goals = np.delete(goals, filter_index, 0)
    speeds = np.delete(speeds, filter_index, 0)
    actions = np.delete(actions, filter_index, 0)
    logprobs  = np.delete(logprobs, filter_index, 0)
    advs = np.delete(advs, filter_index, 0)
    targets = np.delete(targets, filter_index, 0)
    #-------------------------------------------------------------

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=True)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))



    print('filter {} transitions; update'.format(len(filter_index)))



def build_occupancy_maps(human_states, human_velocities):  # velocity: nonholonomic_robot, human_velocities=holonomic robot
    '''
    param human_states:
    return: tensor of shape
    '''
    occupancy_maps = []

    #other_humans_pose = np.concatenate([np.array([(other_human.px, other_human.py)]) for other_human in human_states[1:]], axis=0)   # except robot self
    #other_humans_vel = np.concatenate([np.array([(other_human.vx, other_human.vy)]) for other_human in human_velocities[1:]], axis=0)   # except robot self
    other_humans_pose = human_states[1:]
    other_humans_vel = human_velocities[1:]
    other_px = other_humans_pose[:,0] - human_states[0][0]
    other_py = other_humans_pose[:,1] - human_states[0][1]
        #print('1st:',other_px, other_py)
    # new x-axis is in the direction of robot's velocity
    robot_velocity_angle = np.arctan2(human_velocities[0][1], human_velocities[0][0])
    other_human_orientation = np.arctan2(other_py, other_px)
    rotation = other_human_orientation - robot_velocity_angle
    distance = np.linalg.norm([other_px, other_py], axis=0)  # [1,2,5,6]
        #print('other_human_orien:',other_human_orientation)
        #print('robot_velocity_angle:',robot_velocity_angle)
        #print('rotation:',rotation)
        #print('distance:',distance)
    other_px = np.cos(rotation) * distance
    other_py = np.sin(rotation) * distance
        #print('2nd:',other_px, other_py)

    # compute indicies of humans in the grid
    cell_size = 1
    cell_num = 4
    om_channel_size = 3
    other_x_index = np.floor(other_px / cell_size + cell_num / 2)   # other_px / 1 + 2
    other_y_index = np.floor(other_py / cell_size + cell_num / 2)
        #print('other_x_index=',other_x_index,other_y_index)
    other_x_index[other_x_index < 0] = float('-inf')
    other_x_index[other_x_index >= cell_num] = float('-inf')   # -inf [0,1,2,3] -inf
    other_y_index[other_y_index < 0] = float('-inf')
    other_y_index[other_y_index >= cell_num] = float('-inf')   # -inf [0,1,2,3] -inf
        #print('refined index=',other_x_index, other_y_index)
    grid_indices = cell_num * other_y_index + other_x_index    # y_index is y-axis, x_index is x-axis
    #print('grid_indicies:',grid_indices)    # each human's indiv. call num. 0(SW) ~ 15(NE), if |x|>2 or |y|>2, then -inf
    occupancy_map = np.isin(range(cell_num ** 2), grid_indices)   # [0,1,2,...,15] range, is there grid_indicies?  T or F
    #print('occupancy_map:',occupancy_map)

    #OM shape: center is robot
    '''
    # Grid Indicies. N: robot's velocity direction
    [15 11  7  3
     14 10  6  2
     13  9  5  1
     12  8  4  0]
    '''

    if om_channel_size == 1:   # just only consider position data
        occupancy_maps.append([occupancy_map.astype(int)])
    else:
        # calculate relative velocity for other agents
        other_human_velocity_angles = np.arctan2(other_humans_vel[:, 1], other_humans_vel[:, 0])   # vy, vx
        rotation = other_human_velocity_angles - robot_velocity_angle
        other_vx = other_humans_vel[:,0] - human_velocities[0][0]
        other_vy = other_humans_vel[:,1] - human_velocities[0][1]
        speed = np.linalg.norm([other_vx, other_vy], axis=0)

        other_vx = np.cos(rotation) * speed
        other_vy = np.sin(rotation) * speed
        
        #print(other_vx, other_vy)
        dm = [list() for _ in range(cell_num ** 2 * om_channel_size)]    # 4**2 * 3 = 16*3 = 48 (each channel has 16 cells)
        for i, index in np.ndenumerate(grid_indices):
            if index in range(cell_num ** 2):   # range 16
                if om_channel_size == 2:
                    dm[2 * int(index)].append(other_vx[i])
                    dm[2 * int(index) + 1].append(other_vy[i])
                elif om_channel_size == 3:      # maybe pose, vx, and vy??
                    dm[3 * int(index)].append(1)
                    dm[3 * int(index) + 1].append(other_vx[i])
                    dm[3 * int(index) + 2].append(other_vy[i])
                else:
                    raise NotImplementedError
        for i, cell in enumerate(dm):
            dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
        occupancy_maps.append([dm])


    px_list, py_list, vx_list, vy_list = [],[],[],[]


    #print('occupancy map:', occupancy_maps)
    return occupancy_maps
    # return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()