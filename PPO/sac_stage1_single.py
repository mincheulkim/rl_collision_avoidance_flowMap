# https://github.com/CzJaewan/rl_local_planner

import os #test
import logging
import sys
import socket
from tokenize import group
import numpy as np
import rospy #test
import torch #test
import torch.nn as nn #test
import argparse
from mpi4py import MPI #test

from gym import spaces

from torch.optim import Adam #test
from tensorboardX import SummaryWriter
from collections import deque #test


import cv2


from stage_world1 import StageWorld 
from model_sac.sac import SAC, SAC_PED, SAC_MASK, SAC_CCTV, SAC_IROS, SAC_CCTV_HEADER
from model_sac.replay_memory import ReplayMemory

from model.ppo import generate_action_human_sf, human_arrive_goal  # 220708   사람 행동 모델링, 220805 도착여부 판단

import model_sac.clustering as clustering   # 220725
import model_sac.social_zone as social_zone  # 220725




parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Stage",
                    help='Environment name (default: Stage)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')   # HY도 0.99
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')   # HY도 0.005
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')   # HY도 3*10^-4
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter \alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust \alpha (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
#parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
parser.add_argument('--batch_size', type=int, default=512, metavar='N',    # 220818 IROS2021하면서 GPU ram 부족해서
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')   # HY는 500000
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: False)')
parser.add_argument('--laser_beam', type=int, default=512,
                    help='the number of Lidar scan [observation] (default: 512)')
parser.add_argument('--num_env', type=int, default=1,
                    help='the number of environment (default: 10)')
parser.add_argument('--laser_hist', type=int, default=3,
                    help='the number of laser history (default: 3)')
parser.add_argument('--act_size', type=int, default=2,
                    help='Action size (default: 2, translation, rotation velocity)')
parser.add_argument('--epoch', type=int, default=1,
                    help='Epoch (default: 1)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')     
parser.add_argument('--policy_path',  default="single_agent2", 
                    help='policy_path (default: single_agent)')                    
args = parser.parse_args()
# https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md#sac-specific-configurations sac param 참조

robot_visible = False    # 220708

evaluate = False   # 1. 220714
#policy = '' 
#policy = 'ped'     # 2. 220720 ped(SAC-ped) or ''(SAC) or ped_mask(SAC-mask)
#policy = 'ped_mask'   # 220725
policy = 'cctv'   # stage_world1.py에서 1520부분 해제요
#policy = 'cctv_header'   # 220822 cctv_에서 header(cctv 상대위치) 정보 추가한거
#policy = 'IROS2021'  # 220819 as baseline   # args.aprser에서 batch 1024 -> 512로 수정
# for debug ##
LIDAR_visualize = False
mask_visualize = False
CCTV_visualize = False


def run(comm, env, agent, policy_path, args):
    
    test_interval = 10
    #save_interval = 500
    save_interval = 200   # 220715
    #save_interval = 100   # 220819
    # Training Loop
    total_numsteps = 0
    updates = 0

    # world reset
    if env.index == 0: # step
        #Tesnorboard
        writer = SummaryWriter('runs/' + policy_path)
        env.reset_world()

    # replay_memory     
    memory = ReplayMemory(args.replay_size, args.seed)

    avg_reward = 0
    avg_cnt = 0
    
    rule = env.rule            
    #rule = 'group_circle_crossing'  # crossing
    print('시나리오:',rule)

    for i_episode in range(args.num_steps):
    
        episode_reward = 0
        episode_steps = 0
        done = False        
        # Env reset
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
    

        init_poses, init_goals = env.initialize_pose_robot_humans(rule)   # as [[0,0],[0,1],...] and [[1,1],[2,2],...]
        for i, init_pose in enumerate(init_poses):
            env.control_pose_specific(init_pose, i)
        rospy.sleep(1)
                   
        #env.set_init_pose(init_pose)
        env.set_init_goal(init_goals[0])
        
        
        # Get initial state
        frame = env.get_laser_observation()   # frame = obs
        frame_stack = deque([frame, frame, frame])
        goal = np.asarray(env.get_local_goal())
        if policy == 'IROS2021':  # 220819 local goal(x, y, theta)
            goal = np.asarray(env.get_local_goal_three())
        speed = np.asarray(env.get_self_speed())
          
        # 220711 ADDED
        pose_list = env.pose_list
        velocity_list = env.speed_poly_list
        ped = env.get_pedestrain_observation(pose_list, velocity_list)   # 애초에 get_pedestrain_observation에서 ped(as flow map)을 만들어 줘야 할듯
        ## input: state_list, pose_list, velocity_list
        ## output: ped_map (3x60x60)        
        if policy == 'ped':
            ped_stack = deque([ped, ped, ped])
        # 220816 CCTV
        if policy == 'cctv' or policy == 'cctv_header':
            lidar_list = env.lidar_list
            lidar1_stack = deque([lidar_list[1], lidar_list[1], lidar_list[1]])
            lidar2_stack = deque([lidar_list[2], lidar_list[2], lidar_list[2]])
            lidar3_stack = deque([lidar_list[3], lidar_list[3], lidar_list[3]])
            lidar4_stack = deque([lidar_list[4], lidar_list[4], lidar_list[4]])
            lidar5_stack = deque([lidar_list[5], lidar_list[5], lidar_list[5]])
        # 220819 IROS stack
        iros_map = np.concatenate((env.map1, ped), axis=0)
        
        # 220723 group mask layer stack
        if policy == 'ped_mask':
            mask = np.zeros(512)
            mask_stack = deque([mask, mask, mask])
        
        # 220822 get relative pose of cctv            
        if policy == 'cctv_header':
            cctv_pose = env.get_relative_pose_cctv(pose_list)
            
        if policy == 'ped':   # 220720
            state = [frame_stack, goal, speed, ped_stack]
        elif policy =='ped_mask':
            state = [frame_stack, goal, speed, mask_stack]
        elif policy =='cctv':  # 220812
            state = [frame_stack, goal, speed, lidar1_stack, lidar2_stack, lidar3_stack, lidar4_stack, lidar5_stack]
        elif policy =='cctv_header':  # 220822
            state = [frame_stack, goal, speed, lidar1_stack, lidar2_stack, lidar3_stack, lidar4_stack, lidar5_stack, cctv_pose]
        elif policy == 'IROS2021':
            state = [frame_stack, goal, speed, iros_map]
        else:
            state = [frame_stack, goal, speed]
            
        goal_global_list = init_goals   # ADDED
        

        # Episode start
        while not done and not rospy.is_shutdown():                
            
            state_list = comm.gather(state, root=0)
            # state_list(=state) = [frame_stack, goal, speed]
            if policy == 'ped_mask':
                # POSTPROCESSING FOR MAKSED GROUP LABEL based on human information in sensor range
                # [Done] 1. clstering된 pedestrain list 생성(DBSCAN 씀)
                pedestrain_list = clustering.generate_pedestrain_list(env, pose_list, velocity_list)
                # 0: dx, 1:dy, 2:dvx 3:rel.dvy 
                # 4:indiv.id(if detected, else 0) , 5:grp.id(if detected, else 0) , 6:visibility(As 1)
                #print('페드맵:',pedestrain_list)
                # [DONE] 4. Generate individual, group convexhull based on pedestrain_map
                #           and create group_mask_layer
                mask_layer = social_zone.create_group_mask_layer(env, pedestrain_list)
            
                '''
                # lidar sensor data 및 mask 로깅
                logger_lidar.info('{}'.format(frame_stack[2]))
                logger_mask.info('{}'.format(mask_stack[2]))
                '''
            
            
            # Robot action
            if env.index == 0:                
                if evaluate:
                        action = agent.select_action(state_list, evaluate=True)
                else:
                        action = agent.select_action(state_list)  # Sample action from policy
            else:
                action = None
                
            # Generate human actions
            robot_state = state_list[0:1]   # 211126 https://jinisbonusbook.tistory.com/32   # ADDED
            pose_list = np.array(pose_list) # ADDED
            speed_poly_list = env.speed_poly_list   # ADDED
            speed_poly_list =np.array(speed_poly_list)   # ADDED
            human_actions, scaled_position=generate_action_human_sf(env=env, pose_list=pose_list[:,0:2], goal_global_list=goal_global_list, num_env=num_human, robot_visible=robot_visible, grp_list=env.human_list, scenario=env.scenario)   # ADDED

            if env.index == 0:
                if evaluate:
                    pass
                else:
    
                    if len(memory) > args.batch_size:
                        # Number of updates per step in environment
                        for i in range(args.updates_per_step):    # TODO : 늘려봐야겠다   # TODO sac 최신버전 entropy 학습유무(target entropy) action의 디멘젼
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)  # batch_size = 1024, updates = 1

                            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                            writer.add_scalar('loss/policy', policy_loss, updates)
                            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                            updates += 1
                    
            # Execute actions
            #-------------------------------------------------------------------------            
        
            '''
            real_action = comm.scatter(action, root=0)    
            
            #real_action[np.isnan(real_action)] = 0.1
            #real_action[np.isnan(real_action)] = 0.1
            #print(real_action)
            env.control_vel(real_action)
            '''
            real_action = comm.scatter(action, root=0)    
            
            for i in range(num_human):
                if i==0:
                    env.control_vel_specific(real_action, i)
                    #env.control_vel_specific([1.0,0], i)
                #elif i==1:   # DEBUG 특정 i번째 사람
                #    env.control_vel_specific([1.0, 0.0], i)
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
                    if policy == 'cctv' or policy == 'cctv_header' or policy == 'IROS2021':
                        pass
                    else:
                        env.control_vel_specific(scaled_action, i)
                        #env.control_pose_specific(init_poses[i], i)
            rospy.sleep(0.001)
            
             
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
                    #lidar = cv2.applyColorMap(255-lidar, cv2.COLORMAP_JET)
                lidar = cv2.resize(lidar, dsize=(512,256), interpolation=cv2.INTER_NEAREST)   # ColorMap flag: https://076923.github.io/posts/Python-opencv-8/
                cv2.imshow("Local flow map", lidar)
                cv2.waitKey(1)    
                
            # CCTV visualize, 3 * 512 2D LIDAR history map  # 211220
            if CCTV_visualize:
                #cctv = np.stack(((lidar3_stack[0])*255, (lidar3_stack[1])*255, (lidar3_stack[2])*255), axis=0) 
                cctv = np.stack(((lidar3_stack[0]+0.5)*255, (lidar3_stack[1]+0.5)*255, (lidar3_stack[2]+0.5)*255), axis=0) 
                            # 여기 1 3개를 2~5로 바꿔주면 됨
                cctv = np.uint8(cctv)
                    #lidar = cv2.applyColorMap(lidar, cv2.COLORMAP_BONE)
                    #lidar = cv2.applyColorMap(255-lidar, cv2.COLORMAP_JET)
                cctv = cv2.resize(cctv, dsize=(512,256), interpolation=cv2.INTER_NEAREST)   # ColorMap flag: https://076923.github.io/posts/Python-opencv-8/
                cv2.imshow("CCTV1", cctv)
                cv2.waitKey(1)   

            ## Get reward and terminal state
            reward, done, result = env.get_reward_and_terminate(episode_steps)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Get next state
            next_frame = env.get_laser_observation()
            left = frame_stack.popleft()
            frame_stack.append(next_frame)
            
            if policy == 'cctv' or policy == 'cctv_header': 
                # Get next lidars # 220812
                lidar_list = env.lidar_list
                left = lidar1_stack.popleft()
                lidar1_stack.append(lidar_list[1])
                left = lidar2_stack.popleft()
                lidar2_stack.append(lidar_list[2])
                left = lidar3_stack.popleft()
                lidar3_stack.append(lidar_list[3])
                left = lidar4_stack.popleft()
                lidar4_stack.append(lidar_list[4])
                left = lidar5_stack.popleft()
                lidar5_stack.append(lidar_list[5])
            
            # Get next group mask # 220725
            # update mask_Stack
            pedestrain_list_new = clustering.generate_pedestrain_list(env, env.pose_list , env.speed_poly_list)
            if policy == 'ped_mask':
                mask_layer = social_zone.create_group_mask_layer(env, pedestrain_list_new)
                left_r = mask_stack.popleft()
                mask_stack.append(mask_layer)        
            
            # 220711 ADDED Pedestrain map visualize
            # ped_stack: 왼쪽부터 빠짐 (t-2, t-1, t) 가장 오른쪽에 append[2]된게 최신
                           # 앞칼럼: 시간(t-2, t-1, t), 뒷칼럼(pose, vx, vy)
            #dist = cv2.resize(ped_stack[2][0], dsize=(600,600))#, interpolation=cv2.INTER_LINEAR)  vx
            #dist2 = cv2.resize(ped_stack[2][1], dsize=(480,480))#, interpolation=cv2.INTER_LINEAR)  vy
            #dist3 = cv2.resize(ped_stack[2][2], dsize=(480,480))#, interpolation=cv2.INTER_LINEAR)
            #cv2.imshow("Local flow map1", dist)
            #cv2.imshow("Local flow map2", dist2)
            #cv2.imshow("Local flow map3", dist3)
            #cv2.waitKey(1)
            
            # 220711 ADDED
            pose_list = env.pose_list
            velocity_list = env.speed_poly_list
            next_ped = env.get_pedestrain_observation(pose_list, velocity_list)
            if policy == 'ped':
                left = ped_stack.popleft()
                ped_stack.append(next_ped)
            
            next_goal = np.asarray(env.get_local_goal())
            # 220819
            if policy == 'IROS2021':
                next_goal = np.asarray(env.get_local_goal_three())
            next_speed = np.asarray(env.get_self_speed())
            
            # iros_map 갱신 220819
            next_iros_map = np.concatenate((env.map1, next_ped), axis=0)
            
            # 220822 CCTV_header(rel cctv pose) 갱신
            if policy == 'cctv_header':
                cctv_pose = env.get_relative_pose_cctv(pose_list)
            
            if policy == 'ped':   # 220720
                next_state = [frame_stack, next_goal, next_speed, ped_stack]  # 220712
            elif policy == 'ped_mask':
                next_state = [frame_stack, next_goal, next_speed, mask_stack]  # 220712
            elif policy == 'cctv':  # 220812
                next_state = [frame_stack, next_goal, next_speed, lidar1_stack, lidar2_stack, lidar3_stack, lidar4_stack, lidar5_stack]  # 220812
            elif policy == 'cctv_header':  # 220822
                next_state = [frame_stack, next_goal, next_speed, lidar1_stack, lidar2_stack, lidar3_stack, lidar4_stack, lidar5_stack, cctv_pose]  # 220812
            elif policy == 'IROS2021':   # 220819
                next_state = [frame_stack, next_goal, next_speed, next_iros_map]  # 220819
            else:
                next_state = [frame_stack, next_goal, next_speed]

            r_list = comm.gather(reward, root=0)
            done_list = comm.gather(done, root=0)
            next_state_list = comm.gather(next_state, root=0)
            
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            
            if env.index == 0:
                #meomry.list_push(state_list, action, r_list, next_state_list, done_list)
                #for i in range(np.asarray(state_list).shape[0]):
                for i in range(np.asarray(state_list, dtype=object).shape[0]):  # 220711 https://stackoverflow.com/questions/63097829/debugging-numpy-visibledeprecationwarning-ndarray-from-ragged-nested-sequences
                    if policy == 'ped':   # 220720
                        memory.push_ped(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i],  # Append transition to memory
                                next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i], 
                                state_list[i][3], next_state_list[i][3])  # added ped_map, next_ped_map  220712
                    elif policy == 'ped_mask':
                         memory.push_mask(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i],  # Append transition to memory
                                next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i], 
                                state_list[i][3], next_state_list[i][3])  # added mask, next_mask 220725
                    elif policy == 'cctv':  # 220812
                        memory.push_cctv(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i], 
                                         next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i], 
                                         state_list[i][3], state_list[i][4], state_list[i][5], state_list[i][6], state_list[i][7], 
                                         next_state_list[i][3], next_state_list[i][4], next_state_list[i][5], next_state_list[i][6], next_state_list[i][7]) 
                    elif policy == 'cctv_header':  # 220822
                        memory.push_cctv_header(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i], 
                                         next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i], 
                                         state_list[i][3], state_list[i][4], state_list[i][5], state_list[i][6], state_list[i][7], 
                                         next_state_list[i][3], next_state_list[i][4], next_state_list[i][5], next_state_list[i][6], next_state_list[i][7], 
                                         next_state_list[i][8])   # cctv_header
                    elif policy == 'IROS2021':   # 22819
                        memory.push_iros(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i],  # Append transition to memory
                                next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i], 
                                state_list[i][3], next_state_list[i][3])  # 220819 iros map
                    else:  # 220720 (오리지널)
                        memory.push(state_list[i][0], state_list[i][1], state_list[i][2], action[i], r_list[i], 
                                next_state_list[i][0], next_state_list[i][1], next_state_list[i][2], done_list[i]) # Append transition to memory


            state = next_state  
            
            if mask_visualize and policy == 'ped_mask':
                ## 220725 mask visualize
                # mask visualize
                masking = np.stack((255-(mask_stack[0])*255, 255-(mask_stack[1])*255, 255-(mask_stack[2])*255), axis=0)   # RGB?
                masking = np.uint8(masking)

                masking = cv2.resize(masking, dsize=(512,256), interpolation=cv2.INTER_NEAREST)   # ColorMap flag: https://076923.github.io/posts/Python-opencv-8/
                cv2.imshow("Local masking map", masking)
                cv2.waitKey(1) 
                
            ## 220805 check human arrived at the init goal
            human_arrival_list = human_arrive_goal(pose_list, goal_global_list, num_human)
            #print('도착리스트:',human_arrival_list)  # [[0][1][0][0][0][0]]
            for i, arrival in enumerate(human_arrival_list):
                if arrival == 1 and i!=0:
                    #print(i,'번째는 도착했습니다',arrival)
                    goal_global_list[i] = init_poses[i][:2]

        if total_numsteps > args.num_steps:
            break        
        
        #220812 end after 100 episode
        #if i_episode > 100:
        #    break
        
        avg_cnt += 1

        if env.index == 0:
            writer.add_scalar('reward/train', episode_reward, i_episode)
            if i_episode != 0 and i_episode % save_interval == 0:
    
                torch.save(agent.policy.state_dict(), policy_path + '/policy_epi_{}'.format(i_episode))
                print('########################## policy model saved when update {} times#########'
                            '################'.format(i_episode))
                torch.save(agent.critic_1.state_dict(), policy_path + '/critic_1_epi_{}'.format(i_episode))
                print('########################## critic model saved when update {} times#########'
                            '################'.format(i_episode))
                torch.save(agent.critic_2.state_dict(), policy_path + '/critic_2_epi_{}'.format(i_episode))
                print('########################## critic model saved when update {} times#########'
                            '################'.format(i_episode))
                
                ## 220825 save_dict 추가     # https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
                torch.save(agent.critic_1_target.state_dict(), policy_path + '/critic_1_target_epi_{}'.format(i_episode))
                torch.save(agent.critic_2_target.state_dict(), policy_path + '/critic_2_target_epi_{}'.format(i_episode))
                torch.save(agent.policy_optim.state_dict(), policy_path + '/policy_optim_epi_{}'.format(i_episode))
                torch.save(agent.critic_1_optim.state_dict(), policy_path + '/critic_1_optim_epi_{}'.format(i_episode))
                torch.save(agent.critic_2_optim.state_dict(), policy_path + '/critic_2_optim_epi_{}'.format(i_episode))
                
                


        #print("Episode: {}, episode steps: {}, reward: {}, result: {}".format(i_episode, episode_steps, round(reward, 2), result))
        print("Episode: {}, episode steps: {}, reward: {}, result: {}".format(i_episode, episode_steps, episode_reward, result))  # 220711. 1.round제거 2.reward->episode_reward

        #avg_reward += round(episode_reward, 2)
        avg_reward += episode_reward  # 220711
        if avg_cnt % 100 == 0:
            print("Average reward: {}".format(avg_reward/avg_cnt))
            writer.add_scalar('Avg.reward/100', avg_reward/avg_cnt, avg_cnt/100)   # ADDED    



if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    '''
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
    '''
    
    #220727 Logging Lidar and mask
    '''
    lidar_file = './log/' + hostname + '/lidar.log'
    mask_file = './log/' + hostname + '/mask.log'
    logger_lidar = logging.getLogger('loggerlidar')
    logger_mask = logging.getLogger('loggermask')
    logger_lidar.setLevel(logging.INFO)
    logger_mask.setLevel(logging.INFO)
    lidar_file_handler = logging.FileHandler(lidar_file, mode='a')
    mask_file_handler = logging.FileHandler(mask_file, mode='a')
    lidar_file_handler.setLevel(logging.INFO)
    mask_file_handler.setLevel(logging.INFO)
    logger_lidar.addHandler(lidar_file_handler)
    logger_mask.addHandler(mask_file_handler)
    '''

    comm = MPI.COMM_WORLD    # instantize the communication world
    rank = comm.Get_rank()   # get this particular processes' `rank` ID
    size = comm.Get_size()   # get the size of the communication world
    # PID = os.getpid()
    # Check backward
    print("MPI size=%d, rank=%d" % (size, rank))

    # Environment
    env = StageWorld(beam_num=args.laser_beam, index=rank, num_env=args.num_env)   # laser_beam = 512, rank=0?, num_env = 1
    print("Ready to environment")
    policy_path = args.policy_path
    
    reward = None
    action_bound = [[0, -1], [1, 1]] ####
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        # Agent num_frame_obs, num_goal_obs, num_vel_obs, action_space, args
        action_bound = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        print("action_bound.shape: ", action_bound.shape, "action_bound", action_bound)
        if policy == 'ped':
            agent = SAC_PED(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)   # PPO에서 policy 선언하는 것과 동일
        elif policy == 'ped_mask':
            agent = SAC_MASK(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)   
        elif policy == 'cctv':
            agent = SAC_CCTV(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)   
        elif policy == 'cctv_header':
            agent = SAC_CCTV_HEADER(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)   
        elif policy == 'IROS2021':
            agent = SAC_IROS(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)
        else:
            agent = SAC(num_frame_obs=args.laser_hist, num_goal_obs=2, num_vel_obs=2, action_space=action_bound, args=args)   # PPO에서 policy 선언하는 것과 동일

        if not os.path.exists(policy_path):   # policy_path  PPO/single_agent2
            os.makedirs(policy_path)

        # Load specific policy
        file_policy = policy_path + '/policy_epi_2200'   
        file_critic_1 = policy_path + '/critic_1_epi_2200'
        file_critic_2 = policy_path + '/critic_2_epi_2200'
        # 0825
        file_critic_1_target = policy_path + '/critic_1_target_epi_2200'
        file_critic_2_target = policy_path + '/critic_2_target_epi_2200'
        file_policy_optim = policy_path + '/policy_optim_epi_2200'
        file_critic_1_optim = policy_path + '/critic_1_optim_epi_2200'
        file_critic_2_optim = policy_path + '/critic_2_optim_epi_2200'
        #복구

        if os.path.exists(file_policy):
            #logger.info('###########################################')
            #logger.info('############Loading Policy Model###########')
            print('loading policy model')
            print(file_policy)
            #logger.info('###########################################')
            state_dict = torch.load(file_policy)
            agent.policy.load_state_dict(state_dict)
            ## 220825 따라 밑에 추가 안하고 critic_1,2_target, critic_1,2_optim, policy_optim 불러오게 함
            state_dict_critic1_target = torch.load(file_critic_1_target)
            state_dict_critic2_target = torch.load(file_critic_2_target)
            state_dict_policy_optim = torch.load(file_policy_optim)
            #state_dict_critic_1_optim = torch.load(file_critic_1_optim)
            state_dict_critic_2_optim = torch.load(file_critic_2_optim)
            agent.critic_1_target.load_state_dict(state_dict_critic1_target)
            agent.critic_2_target.load_state_dict(state_dict_critic2_target)
            agent.policy_optim.load_state_dict(state_dict_policy_optim)
            #agent.critic_1_optim.load_state_dict(state_dict_critic_1_optim)
            agent.critic_2_optim.load_state_dict(state_dict_critic_2_optim)
            
            
        else:
            #logger.info('###########################################')
            #logger.info('############Start policy Training###########')
            #logger.info('###########################################')
            print('start policy training')

        if os.path.exists(file_critic_1):
            #logger.info('###########################################')
            #logger.info('############Loading critic_1 Model###########')
            #logger.info('###########################################')
            print('loadinc critic 1')
            state_dict = torch.load(file_critic_1)
            agent.critic_1.load_state_dict(state_dict)
        else:
            #logger.info('###########################################')
            #logger.info('############Start critic_1 Training###########')
            #logger.info('###########################################')
            print('start critic 1')
    
        if os.path.exists(file_critic_2):
            #logger.info('###########################################')
            #logger.info('############Loading critic_2 Model###########')
            #logger.info('###########################################')
            print('loading critic 2')
            state_dict = torch.load(file_critic_2)
            agent.critic_2.load_state_dict(state_dict)
        else:
            #logger.info('###########################################')
            #logger.info('############Start critic_2 Training###########')
            #logger.info('###########################################')    
            print('start critic 2')

    else:   # rank != 0인것들 신경 x
        agent = None
        policy_path = None
        
    print('폴리시:',policy)

    try:
        print("run")
        run(comm=comm, env=env, agent=agent, policy_path=policy_path, args=args)
    except KeyboardInterrupt:
        pass