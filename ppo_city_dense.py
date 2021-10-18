import os
import logging
import sys
import socket
import numpy as np
from numpy.lib.type_check import real
from model.utils import get_goal_point
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model.net import MLPPolicy, CNNPolicy, RVOPolicy, RobotPolicy, RobotPolicy_LM
from stage_city_dense import StageWorld
from model.ppo import ppo_update_city, generate_train_data, generate_train_data_r, ppo_update_city_r
from model.ppo import get_parameters
from model.ppo import generate_action
from model.ppo import transform_buffer, transform_buffer_r

from model.ppo import generate_action_rvo_dense, generate_action_human, generate_action_robot, generate_action_robot_localmap   # 211027

#import model.orca as orcas  # 211020
from tensorboardX import SummaryWriter   # https://github.com/lanpa/tensorboardX/issues/638
# issue when install tensorboardX==1.0.0 -->< class descriptorBase(metaclass=DescriptorMetaclass):  -> (solve)https://www.icode9.com/content-4-1153066.html
# 1) tensorboard --logdir runs/
# 2) google-chrome -> http://desktop-26msce9:6006/

#writer = SummaryWriter()

import matplotlib.pyplot as plt

MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
#HORIZON = 128  # can be 32 ~ 5000                   # TODO increase time horizon?
#HORIZON = 256  # can be 32 ~ 5000                   # TODO increase time horizon?
HORIZON = 384  # can be 32 ~ 5000                   # TODO increase time horizon?
GAMMA = 0.99   # can be 0.99(normal), discount factor
LAMDA = 0.95   # can be 0.9~0.1, Factor for trade-off of bias vs variance of GAE
#BATCH_SIZE = 1024   # can be 4~4096(minibatch?)     # TODO increase batch size?
BATCH_SIZE = 2048   # can be 4~4096(minibatch?)     # TODO increase batch size?
EPOCH = 2   # can be 3~30(number of epoch when optimizing the surrogate loss)
COEFF_ENTROPY = 5e-4   # may be 0~0.01
#COEFF_ENTROPY = 1e-3   # 211102
CLIP_VALUE = 0.1     # can be 0.1, 0.2, 0.3
NUM_ENV = 1  # 211018   # Agents nom
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5   # 0.003 ~ 5e-6

local_map = False
collision_sanity =False

# ERASE NOW!
# for implement SAC
import sac
import models_sac
#batch_size = 64    # need bigger
batch_size = 256    # modified
eval_eps = 10

rl_core = sac.SAC(
    model = [models_sac.PolicyNetGaussian, models_sac.QNet],
    n_actions = 2,
    learning_rate = [0.0001, 0.0001],
    reward_decay = 0.99,
    #memory_size = 10000,
    memory_size = 20000,
    batch_size = batch_size,
    alpha = 0.1,
    auto_entropy_tuning=True)

is_train = True
render = False
load_model = True

model_path = "save/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

if load_model:
    print("Load model ...", model_path)
    rl_core.save_load_model("load", model_path)


# check 1.[city_dense.world] # of human position 2.[ppo_city_dense.py] NUM_ENV, local_map 3. mpiexec -np NUM_ENV

def run(comm, env, policy_r, policy_path, action_bound, optimizer):     # comm, env.stageworld, 'policy', [[0, -1], [1, 1]], adam           from main()
    # rate = rospy.Rate(5)
    buff = []
    buff_r = [] # 211101
    global_update = 0  # for memory update(128)
    global_step = 0   # just for counting total step

    if env.index == 0:
        env.reset_world()    #    # reset stage, self speed, goal, r_cnt, time

    # for SAC
    total_step = 0
    max_success_rate = 0
    success_count = 0

    for id in range(MAX_EPISODES):    # 5000   # refresh for a agent
        
        #env.reset_pose()   # reset initial pose(x,y,theta)
        #env.generate_goal_point()   # generate global goal & local goal

        env.generate_pose_goal_circle()  # shafeshift above two line

        terminal = False
        ep_reward = 0
        live_flag = True
        step = 1  # is used for Time Out(limit 150)

        obs = env.get_laser_observation()   # e.g. array([0.5, 0.5, ...., 0.24598769, 24534221]), total list length 512
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())   # local goal: perspective of robot coordinate
        speed = np.asarray(env.get_self_speed())
        
        speed_poly = np.asarray(env.get_self_speed_poly())  # 211103
        pose_ori = env.get_self_stateGT()   # 211019
        pose = np.asarray(pose_ori[:2])   # 211019
        rot = np.asarray(pose_ori[2])

        state = [obs_stack, goal, speed]  # state: [deque([array([]),array([]),array([])]), array([-0.2323, 8.23232]), array[0, 0]]    # 3 stacted 512 lidar, local goal, init speed

        goal_global = np.asarray(env.get_goal_point())

        # for sac
        loss_a = loss_c = 0.
        acc_reward = 0.
        
        
        while not terminal and not rospy.is_shutdown():   # terminal is similar as info(done)
            state_list = comm.gather(state, root=0)   # incorporate observation state
            pose_list = comm.gather(pose, root=0)     # 211019. 5 states for each human
            velocity_poly_list = comm.gather(speed_poly, root=0)  # 211104, for [vx,vy]
            goal_global_list = comm.gather(goal_global, root=0)
            rot_list = comm.gather(rot, root=0)       # 211108. robot's rotation (-180~180 U, -180~180 D)

            # 1. generate actions            
            # for human 211002
            scaled_action=generate_action_human(env=env, state_list=state_list, pose_list=pose_list, goal_global_list=goal_global_list, num_env=NUM_ENV)   # from orca, 21102        

            # for robot  211101
            if local_map:   # generate_action_human with local_flowmap
                v_r, a_r, logprob_r, scaled_action_r, occupancy_maps_r=generate_action_robot_localmap(env=env, state=state, pose=pose, policy=policy_r, action_bound=action_bound, state_list=state_list, pose_list=pose_list, velocity_poly_list=velocity_poly_list, evaluate=False)  # for training
            else:   # baseline RL policy
                v_r, a_r, logprob_r, scaled_action_r=generate_action_robot(env=env, state=state, pose=pose, policy=policy_r, action_bound=action_bound, evaluate=False)  # for training
                #v_r, a_r, logprob_r, scaled_action_r=generate_action_robot(env=env, state=state, pose=pose, policy=policy_r, action_bound=action_bound, evaluate=true)  # for test

            # for sac
            if env.index ==0:
                if is_train:
                    action = rl_core.choose_action(state, eval=False)
                else:
                    action = rl_core.choose_action(state, eval=True)

            # 2. execute actions
            # human part
            real_action = comm.scatter(scaled_action, root=0)  # discretize scaled action   e.g. array[0.123023, -0.242424]  seperate actions and distribue each env
            if live_flag:
                if env.index ==0:   # robot
                    # for sac
                    scaled_action = np.clip(action, a_min=action_bound[0], a_max=action_bound[1])
                    env.control_vel(scaled_action)
                    #env.control_vel(scaled_action_r)  # original
                else:  # TODO check rvo vel, humans
                    angles = np.arctan2(real_action[1], real_action[0])
                    diff = angles - rot
                    length = np.sqrt([real_action[0]**2+real_action[1]**2])
                    mod_vel = (length, diff)
                    env.control_vel(mod_vel)   # 211108
            # rate.sleep()
                rospy.sleep(0.001)

            # 3. get informtion after action(reward, info)
                r, terminal, result = env.get_reward_and_terminate(step)   # for each agents(run like dummy_vec). # float, T or F, description(o is base)
                #print(env.index,'s termination: ',terminal, result)  #erase
                ep_reward += r   # for one episode culm reward
                step += 1   # time goes on +1
                #print(step,' reward: ',r, 'ep_reard:',ep_reward)

            # 3.1 check collision via sanity check
                # my position
            if env.index==0 and collision_sanity:
                #print(env.index,'s pose:',pose)
                for i in range(1, NUM_ENV):  # human 1~NumEnv
                    distance = np.sqrt((pose_list[0][0]-pose_list[i][0])**2+(pose_list[0][1]-pose_list[i][1])**2)       # distance = my pose - other pose
                    #print(pose_list[i], i, distance)                # other positions
                    if distance <=1.15:                         # distance < 0
                        #print('whhoray!!')
                        terminal = True
                        result = 'Crashed'
                        ep_reward -= r
                        r = -15.
                        ep_reward += r

            if terminal==True:
                live_flag=False
                if result == 'Reach Goal':
                    success_count +=1

            global_step += 1   # 0 to add 1   # always increase(do not regard reset env)

            # 4. get next state
            s_next = env.get_laser_observation()   # get new lidar obs
            left = obs_stack.popleft()   # remove left stack(3 consequence data use)  # obs_stack:deque[obs, obs, obs], left = trash(don't use)
            obs_stack.append(s_next)     # add right data to stack
            goal_next = np.asarray(env.get_local_goal())   # get updated local goal based on changed agent state
            speed_next = np.asarray(env.get_self_speed())  # ???
            state_next = [obs_stack, goal_next, speed_next]    # original state declare like 'state = [obs_stack, goal, speed]'
                                                               # get (updated l-r+) obs_stack, new local goal and updated speed_next
            # 4.1 get next state(pose, vel)
            pose_ori_next = env.get_self_stateGT()   # 211019
            pose_next = np.asarray(pose_ori_next[:2])   # 211019
            speed_next_poly = np.asarray(env.get_self_speed_poly())  # 211103
            rot_next = np.asarray(pose_ori_next[2])   # 211108

            if global_step % HORIZON == 0:   # every 128, estimate future V???     
                if local_map:   # localMap
                    last_v_r, _, _, _, _=generate_action_robot_localmap(env=env, state=state_next, pose=pose, policy=policy_r, action_bound=action_bound, state_list=state_list, pose_list=pose_list, velocity_poly_list=velocity_poly_list, evaluate=False)  # for training
                else:
                    last_v_r, _, _, _=generate_action_robot(env=env, state=state_next, pose=pose_next, policy=policy_r, action_bound=action_bound, evaluate=False)  # training
                    #last_v_r, _, _, _=generate_action_robot(env=env, state=state_next, pose=pose_next, policy=policy_r, action_bound=action_bound, evaluate=True)  # test

            # for sac, add transition
            if env.index ==0:
                end = 0 if terminal else 1
                rl_core.store_transition(state, action, r, state_next, end)
                # learn the model
                loss_a = loss_c = 0.
                if total_step > batch_size and is_train:
                    loss_a, loss_c = rl_core.learn()
                total_step += 1

                # print information
                acc_reward += r
                #print(id, step, total_step, action, r, loss_a, loss_c, rl_core.alpha, acc_reward/step)
                
                
                print('\rEps:{:3d} /{:4d} /{:6d}| action:{:+.2f}| R:{:+.2f}| Loss:[A>{:+.2f} C>{:+.2f}]| Alpha: {:.3f}| Ravg:{:.2f}| Cum_avg:{:.3f}  '\
                    .format(id, step, total_step, action[0], r, loss_a, loss_c, rl_core.alpha, acc_reward/step, acc_reward))
                
                
                
            # 5. add transitons in buff and update policy
            if env.index == 0:  # maybe env.index=0 means robot
                # TODO. state, a, r_list, terminal_list, logprob, v only cares robot[0], num_env = 1
                if local_map:
                    buff_r.append((state, a_r, r, terminal, logprob_r, v_r, occupancy_maps_r))   # for robot buffer
                else:
                    buff_r.append((state, a_r, r, terminal, logprob_r, v_r))   # for robot buffer
                # 3 stacked lidar+relative dist+vel, [[1.23,232],...,[1.123,2.323] #5], [0.212, ... 3 ..., 0.112], [F, F, F, F, F], [-2.232, ..., 02.222], [-0.222, ..., -0.222]
                #                  state                                                         r_list           terminal_list         logprob                   v
                #print('len buff_r:',len(buff_r))
                if len(buff_r) > HORIZON - 1:   # FOR ROBOT
                    if local_map:
                        s_batch_r, goal_batch_r, speed_batch_r, a_batch_r, r_batch_r, d_batch_r, l_batch_r, v_batch_r, occupancy_maps_r = \
                        transform_buffer_r(buff_r=buff_r, LM=True)   # from model.ppo, batched buffer
                        t_batch_r, advs_batch_r = generate_train_data_r(rewards=r_batch_r, gamma=GAMMA, values=v_batch_r,  # r_batch, 0.99, v_batch
                                                              last_value=last_v_r, dones=d_batch_r, lam=LAMDA)   # last_v(every 128, future v), terminal list, 0.95
                        memory_r = (s_batch_r, goal_batch_r, speed_batch_r, a_batch_r, l_batch_r, t_batch_r, v_batch_r, r_batch_r, advs_batch_r, occupancy_maps_r)
                        ppo_update_city_r(policy=policy_r, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory_r,  # CNNPolicy, Adam, 1024, above lie about memory
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,  # 2, 5e-4, 0.1, 128
                                            num_env=1, frames=LASER_HIST,   # 20, 3
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE, LM=True)   # 512, 2
                    else:
                        s_batch_r, goal_batch_r, speed_batch_r, a_batch_r, r_batch_r, d_batch_r, l_batch_r, v_batch_r = \
                        transform_buffer_r(buff_r=buff_r, LM=False)   # from model.ppo, batched buffer
                        t_batch_r, advs_batch_r = generate_train_data_r(rewards=r_batch_r, gamma=GAMMA, values=v_batch_r,  # r_batch, 0.99, v_batch
                                                              last_value=last_v_r, dones=d_batch_r, lam=LAMDA)   # last_v(every 128, future v), terminal list, 0.95
                        memory_r = (s_batch_r, goal_batch_r, speed_batch_r, a_batch_r, l_batch_r, t_batch_r, v_batch_r, r_batch_r, advs_batch_r)   # before occupancy maps
                        ppo_update_city_r(policy=policy_r, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory_r,  # CNNPolicy, Adam, 1024, above lie about memory
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,  # 2, 5e-4, 0.1, 128
                                            num_env=1, frames=LASER_HIST,   # 20, 3
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE, LM=False)   # 512, 2

                    buff_r = []          # clean buffer
                    global_update += 1   # counting how many buffer transition and cleaned(how many time model updated)

            #step += 1   # time goes on +1
            state = state_next
            
            pose = pose_next   # 2l.,j,j,11020
            speed_poly = speed_next_poly  # 211104
            rot = rot_next

        # after terminate = True(end step)
        #distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        
        if env.index ==0:   # log save
            logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode(id) %05d, stepp %03d, Reward %-5.1f, Result: %s' % \
                        (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, result))
            logger_cal.info(ep_reward)

            #if id != 0 and id % 20 == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy_r.state_dict(), policy_path + '/Robot_r_{}_step'.format(id))   # save pth at every 20th model updated
                logger.info('########################## model saved when update {}global times and {} steps, {} episode#########'
                            '################'.format(global_update, step, id))
            
        if id > 0 and id % eval_eps ==0:
            # Success rate
            success_rate = success_count / eval_eps
            success_count = 0
            # save the best model
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                if is_train:
                    print("Save wiston SAC model to "+model_path)
                    rl_core.save_load_model("save", model_path)
                print("Success Rate (current/max):", success_rate, "/", max_success_rate)



        # setting tips for ppo: https://github.com/Unity-Technologies/ml-agents/blob/main/docs/localized/KR/docs/Training-PPO.md



if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()  # e.g. DESKTOP-26MSCE9
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'   # save output.log
    cal_file = './log/' + hostname + '/cal.log'         # save cal.log

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
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(512, index=rank, num_env=NUM_ENV)   # 512(obs_size as lidar size), index, 5
    
    reward = None
    action_bound = [[0, -1], [1, 1]]
    
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:   # (env.index=0))
        policy_path = 'policy'
        if local_map:
            policy_r = RobotPolicy_LM(frames=LASER_HIST, action_space=2)   # 211104 robot with lm
        else:
            policy_r = RobotPolicy(frames=LASER_HIST, action_space=2)   # 211104 robot with lm
        
        policy_r.cuda()
        opt = Adam(policy_r.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):   # 'policy'
            os.makedirs(policy_path)

        file_r = policy_path + '/final.pth'
        #file_r = policy_path + '/Robot_r_3260_step.pth'

        print('current Robot policy:',policy_r)
        if os.path.exists(file_r):
            logger.info('####################################')
            logger.info('#########Loading Robot Model########')
            logger.info(file_r)   # which file is loaded
            logger.info('####################################')
            state_dict = torch.load(file_r)
            policy_r.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy_r = None  # 211101. robot's policy
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy_r=policy_r, policy_path=policy_path, action_bound=action_bound, optimizer=opt)   # comm, env.stageworld, 'policy', [[0, -1], [1, 1]], adam
    except KeyboardInterrupt:
        pass


