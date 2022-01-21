import enum
import os #test
import logging
import sys
import socket
import numpy as np
from torch.optim.optimizer import Optimizer #test
import rospy #test
import torch #test
import torch.nn as nn #test

import cv2

import pickle  # 211215 for save buffer

from mpi4py import MPI #test

from torch.optim import Adam #test
from collections import deque #test

from model.net import MLPPolicy, CNNPolicy, LM_Policy, stacked_LM_Policy, concat_LM_Policy #test
from stage_world1 import StageWorld #test
from model.ppo import ppo_update_stage1, generate_train_data, ppo_update_stage1_stacked_LM, ppo_update_stage1_LM  # 211214
from model.ppo import generate_action, generate_action_human, generate_action_human_groups, generate_action_human_sf, generate_action_LM, generate_action_stacked_LM, generate_action_concat_LM
from model.ppo import transform_buffer, transform_buffer_stacked_LM # 211214 #test
from dbscan.dbscan import DBSCAN
from dbscan.dbscan_new import DBSCAN_new

# 에발류에이션  1.Max Episode 5000->500  2. test_policy False->True  3.SEED 1234 -> 4321
MAX_EPISODES = 5000   # For Train    5000
#MAX_EPISODES = 500     # For Test
LASER_BEAM = 512
LASER_HIST = 3


HORIZON = 1024    # v3            # 220111 TODO as 2048
#HORIZON = 512

GAMMA = 0.99
LAMDA = 0.95
#BATCH_SIZE = 1024   # oriignal
BATCH_SIZE = 128   # is small batch is good? 64?   # 220105 메모리 모잘라서 1024/32 = 32   
#BATCH_SIZE = 32
#TODO SGD style learning에서는 매개변수가 작을수록 더 잘된다고 하네...

EPOCH = 2
COEFF_ENTROPY = 5e-4
#CLIP_VALUE = 0.1
CLIP_VALUE = 0.2
NUM_ENV = 1     # worlds/Group_circle.world
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5

LM_visualize = False    # True or False         # visualize local map(s)
DBSCAN_visualize=False
LIDAR_visualize = False    # 3 row(t-2, t-1, t), rows(512) => 3*512 2D Lidar Map  to see interval t=1 is available, what about interval t=5
policy_list = 'concat_LM'      # select policy. [LM, stacked_LM, '', concat_LM]
robot_visible = False           # 220118
test_policy=False      # For test:True, For Train: False(default)
#test_policy=True      # For test:True, For Train: False(default)


# For fixed Randomization  211230
import random
'''
SEED = 1234  # for training
#SEED = 4321 # for test
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
'''


def run(comm, env, policy, policy_path, action_bound, optimizer):
    # rate = rospy.Rate(5)
    buff = []

    global_update = 0
    global_step = 0
    memory_size = 0


    env.reset_world()
    
    avg_success_nav_time=0
    avg_success_nag_length=0
    avg_success_min_dist=0.0
    avg_success_min_inclusion = 0
    success_counter = 0.00000000001
    
    # 220121
    

    for id in range(MAX_EPISODES):
        terminal = False
        ep_reward = 0
        step = 1
        nav_time = 0
        nav_length = 0.0     # 220120 Metric
        min_dist = 999.0     # 220120 Metric
        num_inclusion = 0     # 220120 Metric
        
        # senario reset option
        init_poses = None
        init_goals = None         
        
        num_human = env.num_human
        
        # 220121
        done_list = []
        for i in range(num_human):
            done_list.append(False)
                    
        rule = 'group_circle_crossing'  # crossing
        init_poses, init_goals = env.initialize_pose_robot_humans(rule)   # as [[0,0],[0,1],...] and [[1,1],[2,2],...]
        for i, init_pose in enumerate(init_poses):
            env.control_pose_specific(init_pose, i)
        rospy.sleep(1)
                   
        #env.set_init_pose(init_pose)
        env.set_init_goal(init_goals[0])

        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]
        
        LM = np.zeros((3, 60, 60))
        LM_stack = deque([LM,LM,LM,LM,LM,LM,LM,LM])    # 220105
        #LM_stack = deque([[[[]]],[[[]]],[[[]]],[[[]]],[[[]]],[[[]]],[[[]]],[[[]]]])
        
        speed_poly = np.asarray(env.get_self_speed_poly())  # 211103
        pose_ori = env.get_self_stateGT()   # 211019
        pose = np.asarray(pose_ori[:2])   # 211019
        rot = np.asarray(pose_ori[2])

        goal_global = np.asarray(env.get_goal_point())
        
        # 211228 DBSCAN clustering
        grp_cluster = None
        noise_cluster = None

        while not terminal and not rospy.is_shutdown():
            
            state_list = comm.gather(state, root=0)

            robot_state = state_list[0:1]   # 211126 https://jinisbonusbook.tistory.com/32
        
            pose_list = env.pose_list
            goal_global_list = init_goals
            pose_list = np.array(pose_list)
            
            speed_poly_list = env.speed_poly_list
            speed_poly_list =np.array(speed_poly_list)
                        
            # generate humans action
            human_actions, scaled_position=generate_action_human_sf(env=env, pose_list=pose_list[:,0:2], goal_global_list=goal_global_list, num_env=num_human, robot_visible=robot_visible, grp_list=env.human_list)
            
            # 211228  DBSCAN group clustering
            pose_list_dbscan = pose_list[1:, :-1]
            speed_poly_list_dbscan = speed_poly_list[1:]
            #dbscan = DBSCAN(pose_list_dbscan,2,2)   # init papameters. eps: 클수록 클러스터 사이즈 커짐(클러스터 갯수 감소), 작으면 잡음 포인트 증가. 매우 크게하면 모든 포인트가 하나의 클러스터에 속하게됨
            dbscan = DBSCAN(pose_list_dbscan, speed_poly_list_dbscan,2,2)   # 220119. add relative velocity
            idx,noise = dbscan.run()    # # Run DBSCAN(CLUSTERING)
            g_cluster,n_cluster = dbscan.sort()     # Result SORTING
            #dbscan.plot()          # Visualization results
            #print('기존:',idx)
            
            # New DBSCAN  220119
            dbscan_new = DBSCAN_new(pose_list_dbscan, speed_poly_list_dbscan,2,2)
            labels = dbscan_new.grouping(pose_list_dbscan, speed_poly_list_dbscan)
            #print('새거:',labels)
            idx = labels
            #print(idx)
            # 220121
            for i, done in enumerate(done_list):
                if done:
                    env.control_vel_specific((0,0),i)
                    env.control_pose_specific((init_goals[i][0]+0.01,init_goals[i][1]+0.01,0), i)
            
            
            
            # generate robot action (at rank==0)
            if policy_list=='LM':  # LM: 60x60
                v, a, logprob, scaled_action, LM, LM_stack =generate_action_LM(env=env, state_list=robot_state, pose_list=pose_list[:,0:2], velocity_list=speed_poly_list, policy=policy, action_bound=action_bound, LM_stack=LM_stack, mode=test_policy)                                                                    # env, state_list, pose_list, velocity_poly_list, policy, action_bound
            elif policy_list=='stacked_LM':
                v, a, logprob, scaled_action, LM =generate_action_stacked_LM(env=env, state_list=robot_state, pose_list=pose_list[:,0:2], velocity_list=speed_poly_list, policy=policy, action_bound=action_bound, index=labels, mode=test_policy)
            elif policy_list=='concat_LM':
                v, a, logprob, scaled_action, LM, LM_stack =generate_action_concat_LM(env=env, state_list=robot_state, pose_list=pose_list, velocity_list=speed_poly_list, policy=policy, action_bound=action_bound, LM_stack=LM_stack, index=labels, mode=test_policy)
            else:
                v, a, logprob, scaled_action=generate_action(env=env, state_list=robot_state, policy=policy, action_bound=action_bound, mode=test_policy)
            

            # distribute and execute actions robot and humans
            for i in range(num_human):
                if i==0:
                    env.control_vel_specific(scaled_action, i)
                else:
                    angles = np.arctan2(human_actions[i][1], human_actions[i][0])     # 
                    #diff = angles - rot   # problem
                    diff = angles - pose_list[i,2]
                    # fix SF rotational issus 211222
                    if diff<=-np.pi:
                            #diff = np.pi+(angles-np.pi)-rot
                            diff = (2*np.pi)+diff
                    elif diff>np.pi:
                            #diff = -np.pi-(np.pi+angles)-rot
                            diff = diff - (2*np.pi)
                    
                    length = np.sqrt([human_actions[i][0]**2+human_actions[i][1]**2])
                    scaled_action = (length, diff)   
                    
                    env.control_vel_specific(scaled_action, i)
                    
            # rate.sleep()
            rospy.sleep(0.001)
            
            
            # 211228 Visualize DBSCAN subgroups
            if DBSCAN_visualize:
                img = np.zeros([20,20,3])  # 20 x 20 
                for i, idx in enumerate(idx):    
                    img[10-int(pose_list_dbscan[i,1]),int(pose_list_dbscan[i,0]+10),0]=idx*50 /255.0
                    img[10-int(pose_list_dbscan[i,1]),int(pose_list_dbscan[i,0]+10),1]=100 /255.0
                    img[10-int(pose_list_dbscan[i,1]),int(pose_list_dbscan[i,0]+10),2]=100 /255.0
                    # cyan color: noise(i==0)

                hsv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2HSV)
                hsv=cv2.resize(hsv, dsize=(240,240), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('image',hsv)
                cv2.waitKey(1)
            

            if LM_visualize:
                # if using _LM, delete [0]
                if policy_list == 'LM':
                    dist = cv2.resize(LM, dsize=(480,480), interpolation=cv2.INTER_LINEAR)   # https://076923.github.io/posts/Python-opencv-8/
                    cv2.imshow("Local flow map", dist)
                elif policy_list == 'stacked_LM':
                    dist = cv2.resize(LM[0][0], dsize=(480,480), interpolation=cv2.INTER_LINEAR)
                    dist2 = cv2.resize(LM[0][1], dsize=(480,480), interpolation=cv2.INTER_LINEAR)
                    dist3 = cv2.resize(LM[0][2], dsize=(480,480), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Local flow map", dist)
                    cv2.imshow("Local flow map2", dist2)
                    cv2.imshow("Local flow map3", dist3)
                elif policy_list == 'concat_LM':
                    dist = cv2.resize(LM_stack[7][0], dsize=(480,480), interpolation=cv2.INTER_LINEAR)   # t=0, pose
                    dist2 = cv2.resize(LM_stack[7][1], dsize=(480,480), interpolation=cv2.INTER_LINEAR)  # t=0, vx
                    dist3 = cv2.resize(LM_stack[7][2], dsize=(480,480), interpolation=cv2.INTER_LINEAR)  # t=0, vy
                    cv2.imshow("Local flow map", dist)
                    cv2.imshow("Local flow map2", dist2)
                    cv2.imshow("Local flow map3", dist3)
                cv2.waitKey(1)

            # LIDAR visualize, 3 * 512 2D LIDAR history map  # 211220
            if LIDAR_visualize:
                greyscale = False
                if greyscale:
                    #robot_state[0][0][0](t-2), [0][0][1], [0][0][2](current)
                    lidar = np.stack(((robot_state[0][0][0]+0.5), (robot_state[0][0][1]+0.5), (robot_state[0][0][2]+0.5)), axis=0)   # https://everyday-image-processing.tistory.com/87
                else:
                    lidar = np.stack(((robot_state[0][0][0]+0.5)*255, (robot_state[0][0][1]+0.5)*255, (robot_state[0][0][2]+0.5)*255), axis=0)   # RGB?
                    lidar = np.uint8(lidar)
                    #lidar = cv2.applyColorMap(lidar, cv2.COLORMAP_BONE)
                    lidar = cv2.applyColorMap(255-lidar, cv2.COLORMAP_JET)
                lidar = cv2.resize(lidar, dsize=(512,256), interpolation=cv2.INTER_NEAREST)   # ColorMap flag: https://076923.github.io/posts/Python-opencv-8/
                cv2.imshow("Local flow map", lidar)
                cv2.waitKey(1)          
            
            
            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step, scaled_action)   # 211221 for backward penalty 
            ep_reward += r
            global_step += 1

            # get next states
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            # get next states(addon)
            pose_ori_next = env.get_self_stateGT()   # 211019
            pose_next = np.asarray(pose_ori_next[:2])   # 211019
            speed_next_poly = np.asarray(env.get_self_speed_poly())  # 211103
            rot_next = np.asarray(pose_ori_next[2])   # 211108
            
            ############training#######################################################################################
            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)   # robot state
                
                pose_next_list = env.pose_list     # robot+human state
                pose_next_list = np.array(pose_next_list)
                
                speed_poly_next_list = env.speed_poly_list             # 220105
                speed_poly_next_list =np.array(speed_poly_list)
                
                # DBSCAN one-more
                pose_list_dbscan = pose_next_list[1:, :-1]
                speed_poly_list_dbscan = speed_poly_next_list[1:]

                dbscan_new = DBSCAN_new(pose_list_dbscan, speed_poly_list_dbscan,2,2)
                labels = dbscan_new.grouping(pose_list_dbscan, speed_poly_list_dbscan)

                
                
                
                state_next_list_new = state_next_list[0:1]   # for robot
                if policy_list=='LM':  # LM: 60x60    # 211214
                    last_v_r, _, _, _, _, _ = generate_action_LM(env=env, state_list=state_next_list_new, pose_list=pose_next_list[:,0:2], velocity_list=speed_poly_next_list, policy=policy, action_bound=action_bound, LM_stack=LM_stack, mode=test_policy)
                elif policy_list=='stacked_LM':
                    last_v_r, _, _, _, _ = generate_action_stacked_LM(env=env, state_list=state_next_list_new, pose_list=pose_next_list[:,0:2], velocity_list=speed_poly_next_list, policy=policy, action_bound=action_bound, index=labels, mode=test_policy)
                elif policy_list=='concat_LM':  # LM: 60x60    # 211214
                    last_v_r, _, _, _, _, _ = generate_action_concat_LM(env=env, state_list=state_next_list_new, pose_list=pose_next_list, velocity_list=speed_poly_next_list, policy=policy, action_bound=action_bound, index=labels, LM_stack=LM_stack, mode=test_policy)
                else:
                    last_v_r, _, _, _ = generate_action(env=env, state_list=state_next_list_new, policy=policy, action_bound=action_bound, mode=test_policy)


            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)
            
            r_list_new = r_list[0:1]
            terminal_list_new=terminal_list[0:1]
            
            
            # 220110 reset human pose when collision arise
            for i, crash in enumerate(env.crash_list):
                #print(i, crash, human_actions[i][1], human_actions[i][0])
                if i != 0 and crash == 1:
                    env.control_pose_specific(init_poses[i], i)
                
            if env.index == 0 and not (step == 1 and terminal):
                ############## LM or stacekd LM ######################################################
                if policy_list =='LM':
                    
                    buff.append((robot_state, a, r_list_new, terminal_list_new, logprob, v, LM_stack))   # 211214

                    memory_size += 1

                    if len(buff) > HORIZON - 1:
                        s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch, local_maps_batch = \
                            transform_buffer_stacked_LM(buff=buff)

                        t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                                last_value=last_v_r, dones=d_batch, lam=LAMDA)
                        memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch, local_maps_batch)
                        

                        ppo_update_stage1_LM(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=1, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)   # 211214

                        buff = []
                        global_update += 1
                        
                elif policy_list == 'stacked_LM':
                    buff.append((robot_state, a, r_list_new, terminal_list_new, logprob, v, LM))   # 211214

                    memory_size += 1

                    if len(buff) > HORIZON - 1:
                        s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch, local_maps_batch = \
                            transform_buffer_stacked_LM(buff=buff)

                        t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                                last_value=last_v_r, dones=d_batch, lam=LAMDA)

                        memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch, local_maps_batch)
                        ppo_update_stage1_stacked_LM(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            #num_env=NUM_ENV, frames=LASER_HIST,
                                            num_env=1, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)   # 211214
                    
                        buff = []
                        global_update += 1
                
                elif policy_list =='concat_LM':
                    
                    buff.append((robot_state, a, r_list_new, terminal_list_new, logprob, v, LM_stack))   # 211214

                    memory_size += 1

                    if len(buff) > HORIZON - 1:
                        s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch, local_maps_batch = \
                            transform_buffer_stacked_LM(buff=buff)

                        t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                                last_value=last_v_r, dones=d_batch, lam=LAMDA)
                        memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch, local_maps_batch)
                        

                        ppo_update_stage1_LM(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=1, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)   # 211214

                        buff = []
                        global_update += 1


                ############## original method ######################################################
                else:
                    buff.append((robot_state, a, r_list_new, terminal_list_new, logprob, v))   # new

                    memory_size += 1

                    if len(buff) > HORIZON - 1:
                        s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                            transform_buffer(buff=buff)
                        t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                                last_value=last_v_r, dones=d_batch, lam=LAMDA)
                        memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                        ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                                epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                                num_env=1, frames=LASER_HIST,
                                                obs_size=OBS_SIZE, act_size=ACT_SIZE)
                        buff = []
                        global_update += 1     
                        
            step += 1
            # 220110 Metric as NavLength 추가
            movlength = np.linalg.norm(pose_next-pose)
            nav_length += movlength
            
            # 220110 Metric as minDist, num_inclusion 두개 추가
            proximity = 1.0
            after_pose = env.pose_list
            after_pose = np.array(after_pose)
            for i, pose in enumerate(after_pose):
                diff = pose-after_pose[0]   # 0[0,0], ~, 13[-0.232, -9.2323]
                diff = np.linalg.norm([diff[0],diff[1]])
                #print(i, diff)
                if i!=0 and min_dist > diff:
                    min_dist = diff 
                if i!=0 and diff < proximity:
                    num_inclusion += 1
                    
            # 220110 Metric as zitters
            
            
            ###################################################################################################
            state = state_next
            pose = pose_next   # 2l.,j,j,11020
            speed_poly = speed_next_poly  # 211104
            rot = rot_next
                       
            curr_human_pose_list = env.pose_list     # robot+human state
            curr_human_pose_list = np.array(curr_human_pose_list)
            init_goals_list = np.array(init_goals)
            diff_dist = curr_human_pose_list[:,0:2]-init_goals_list
            #print('curr:',curr_human_pose_list[:,0:2], 'gl_list:',init_goals_list)
            
            diff_dist_length = np.linalg.norm(diff_dist, axis=1)
            #print(diff_dist_length)
            for i, dist in enumerate(diff_dist_length):
                if dist < 0.15 and i != 0:
                    done_list[i] = True
                    
            #print(done_list)
                    

            
        ###### while문 끝 ######
        #####save policy and logger##############################################################################################
        #if global_update != 0 and global_update % 5 == 0:
        if global_update != 0 and global_update % 2 == 0:   # 211217
            #torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
            torch.save(policy.state_dict(), policy_path + '/Stage1')
            #torch.save(policy, policy_path + '/Stage1_{}_tot'.format(global_update))
            torch.save(policy, policy_path + '/Stage1_tot')
            logger.info('########################## model saved when update {} times#########'
                        '################'.format(global_update))
            
        
        if result == 'Reach Goal':
            success_counter += 1     # 1 2 3
            avg_success_nav_time += step
            avg_success_nag_length += nav_length
            avg_success_min_dist += min_dist
            avg_success_min_inclusion += num_inclusion
            
        
        #distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        #distance = 0
        if not (step==2 and terminal):
            logger.info('Env %02d, Goal (%2.2f, %2.2f), Episode %04d, step(NavTime) %03d, Reward %-5.1f, Result %s, Cum.Mem: %05d, NavLength: %2.2f, minDist: %2.3f, numInclus: %03d, avg.suc.nav.time: %3.3f, avg.suc.nav.length: %3.3f, avg.suc.min_Dist: %2.3f, avg.suc.num_Inclu: %2.3f' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, result, memory_size, nav_length, min_dist, num_inclusion, avg_success_nav_time/success_counter, avg_success_nag_length/success_counter, avg_success_min_dist/success_counter, avg_success_min_inclusion/success_counter))
            logger_cal.info(ep_reward)
            
        
                

        ###################################################################################################


if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD    # instantize the communication world
    rank = comm.Get_rank()   # get this particular processes' `rank` ID
    size = comm.Get_size()   # get the size of the communication world
    # PID = os.getpid()
    # Check backward

    env = StageWorld(512, index=rank, num_env=NUM_ENV)

    print("RANK:",rank," ENV")
    buffer = None

    reward = None
    action_bound = [[0, -1], [1, 1]] ####
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        if policy_list == 'stacked_LM':
            policy_path = 'policy'
            # policy = MLPPolicy(obs_size, act_size)
            policy = stacked_LM_Policy(frames=LASER_HIST, action_space=2)
        elif policy_list == 'LM':   # 220105
            policy_path = 'policy'
            policy = LM_Policy(frames=LASER_HIST, action_space=2)
        elif policy_list == 'concat_LM':   # 220105
            policy_path = 'policy'
            policy = concat_LM_Policy(frames=LASER_HIST, action_space=2)
        else:
            policy_path = 'policy'
            # policy = MLPPolicy(obs_size, act_size)
            policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        print(policy)
        policy.cuda()

        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()


        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        # Load model
        file = policy_path + '/Stage1'
        #file = policy_path + '/_____'
        file_tot = policy_path + '/Stage1_ttt'
        #file_tot = policy_path + '/Stage1_5_tot'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('########Loading Model###############')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)

        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')


        if os.path.exists(file_tot):
            logger.info('####################################')
            logger.info('############Loading tot model#######')
            logger.info('####################################')
            policy = torch.load(file_tot)

    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass