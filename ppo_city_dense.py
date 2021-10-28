import os
import logging
import sys
import socket
import numpy as np
from numpy.lib.type_check import real
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model.net import MLPPolicy, CNNPolicy, RVOPolicy
from stage_city_dense import StageWorld
from model.ppo import ppo_update_city, generate_train_data
from model.ppo import get_parameters
from model.ppo import generate_action
from model.ppo import transform_buffer

from model.ppo import generate_action_rvo_dense, generate_action_human   # 211027

#import model.orca as orcas  # 211020
from tensorboardX import SummaryWriter   # https://github.com/lanpa/tensorboardX/issues/638
# issue when install tensorboardX==1.0.0 -->< class descriptorBase(metaclass=DescriptorMetaclass):  -> (solve)https://www.icode9.com/content-4-1153066.html
# 1) tensorboard --logdir runs/
# 2) google-chrome -> http://desktop-26msce9:6006/

writer = SummaryWriter()


 



MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
#NUM_ENV = 24
NUM_ENV = 20  # 211018   # human num
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5



def run(comm, env, policy, policy_path, action_bound, optimizer):     # comm, env.stageworld, 'policy', [[0, -1], [1, 1]], adam           from main()

    # rate = rospy.Rate(5)
    buff = []
    global_update = 0  # for memory update(128)
    global_step = 0   # just for counting total step


    if env.index == 0:
        # 211026. env.index=0 (robot), else: humans
        env.reset_world()    #    # reset stage, self speed, goal, r_cnt, time


    for id in range(MAX_EPISODES):    # 5000   # refresh for a agent
        env.reset_pose()   # reset initial pose(x,y,theta)

        env.generate_goal_point()   # generate global goal & local goal

        terminal = False
        ep_reward = 0
        step = 1  # is used for Time Out(limit 150)

        #print('generate_goal_pose!')
        obs = env.get_laser_observation()   # e.g. array([0.5, 0.5, ...., 0.24598769, 24534221]), total list length 512
        #print('laser scan: ',len(obs))
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())   # local goal: perspective of robot coordinate
        speed = np.asarray(env.get_self_speed())
    
        pose_ori = env.get_self_stateGT()   # 211019
        pose = np.asarray(pose_ori[:2])   # 211019

        #print('pose:',pose)
        state = [obs_stack, goal, speed]  # state: [deque([array([]),array([]),array([])]), array([-0.2323, 8.23232]), array[0, 0]]    # 3 stacted 512 lidar, local goal, init speed
        #print('state:',state)

        # for static goal test   211021
        '''
        print(env,'s goal point:',env.goal_point)
        print(env,'s local goal:',goal)
        print(env,'s pos:',pose_ori)
        '''
        #print(env.index, 'goal:',goal)
        
        while not terminal and not rospy.is_shutdown():   # terminal is similar as info(done)
            state_list = comm.gather(state, root=0)   # incorporate observation state
            pose_list = comm.gather(pose, root=0)     # 211019. 5 states for each human
            #goals_list = comm.gather(goal, root=0)     # 211019
            #print('pose_list:',pose_list)
            #print('goals_list:',goals_list)
            # [array([-6.50951869, -8.42569113]), array([-7.39730693, -8.33040246]), array([-8.64532387, -7.73384512]), array([-8.9156547 , -8.98302644]), array([-8.77155013, -8.20209891])])
            # to rvo2
            # then get action tuples

            # TODO. Create/Update flowMap by robot(agent 0, env.index=0)
            '''
            if env.index == 0:
                if flow_map is None:  # create flowmap
                    #flow_map = [[0,1,1,0,0,0,1,1,0,0],[1,0,1,0,0,1,0,0],[2,3,2,0,0,2,0],...,[2,4,2,0,0,2,0,2,0]]   # like list. # = count of how many human detected?
                    flow_map = get_flowmap(robot_curr_pose,humans_pose,update=false)
                    flow_map = np.array(flow_map)
                else:                 # update flowmap
                    flow_map = get_flowmap(robot_curr_pose,humans_pose,update=true)
                    # flow_map : 12*12 dig0 matrix. axis=1 means the weights? or direction
            '''

            # 1. generate actions at rank==0
            # human part
            # env: <stage_world1.StageWorld instace at 0x7ff758640050> as manything, state_list=[3 stacked 512 lidar, local_goal, self_speed], policy: 'CNNPolicy(~~, action_boud:[[0,-1],[1,1]]
            #print('policy:',policy)
            #v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list, policy=policy, action_bound=action_bound)   # from ppo
            #print('scaled action:',scaled_action)
            # v: array[[-0.112323],[0.2323],[0.123123],[-1.2323],[-0.023232]] like dummy_vec, a: array({[[0.123,0.23],[0.23,0.23],..,[0.232,0.2323]]})  total 5 like dummy_vec, log_prob:[[-2],...,[-2]] as dummy_vec 5, scaled_action: array([[0.232, 0.2323], [0.2323, 0.2323], ..., [0.2323, 0.2323]]) as 5 agent's action
            #print('env:',env, 'state_list:',state_list, 'policy:',policy, 'action_Bound:',action_bound)
            #print('v:',v, 'A:',a, 'logprob:',logprob, 'scaled_action:',scaled_action)

            v, a, logprob, scaled_action=generate_action_human(env=env, state_list=state_list, pose_list=pose_list, policy=policy, action_bound=action_bound)   # from orca, 211020
            '''
                generate_action_rvo_dense(...,+flow_map)
            '''
            #print('pose list:',pose_list)
            #print('goal list:',goal_list)
            #print('scaled action:',scaled_action)

            # TODO. generate actions for robot(env.index=0)
            # only robot whose rank(index)==0 has the state_list which contains other's state.

            # 2. execute actions
            # human part
            real_action = comm.scatter(scaled_action, root=0)  # discretize scaled action   e.g. array[0.123023, -0.242424]  seperate actions and distribue each env
            #print('real action:',real_action)
            env.control_vel(real_action)

            # rate.sleep()
            rospy.sleep(0.001)

            # 3. get informtion after action(reward, info)
            r, terminal, result = env.get_reward_and_terminate(step)   # for each agents(run like dummy_vec). # float, T or F, description(o is base)
            ep_reward += r   # for one episode culm reward
            global_step += 1   # 0 to add 1   # always increase(do not regard reset env)
            #print('step:',step, 'r:',r, 'terminal:',terminal, 'result:',result, 'ep_reward:',ep_reward,'global_step:',global_step)
            #e.g.
            #Env 00 ('step:', 1, 'r:', -0.062560365832449172, 'terminal:', False, 'result:', 0, 'ep_reward:', -0.062560365832449172, 'global_step:', 1)
            #Env 01 ('step:', 1, 'r:', 13.337030629721486, 'terminal:', False, 'result:', 0, 'ep_reward:', 13.337030629721486, 'global_step:', 1)       
            #Env 02 ('step:', 1, 'r:', -0.20109885838264674, 'terminal:', False, 'result:', 0, 'ep_reward:', -0.20109885838264674, 'global_step:', 1)
            #Env 03 ('step:', 1, 'r:', -13.016827513449938, 'terminal:', False, 'result:', 0, 'ep_reward:', -13.016827513449938, 'global_step:', 1)
            #Env 04 ('step:', 1, 'r:', -8.7319593604756527, 'terminal:', False, 'result:', 0, 'ep_reward:', -8.7319593604756527, 'global_step:', 1)


            # 4. get next state
            s_next = env.get_laser_observation()   # get new lidar obs
            left = obs_stack.popleft()   # remove left stack(3 consequence data use)  # obs_stack:deque[obs, obs, obs]
            obs_stack.append(s_next)     # add right data to stack
            goal_next = np.asarray(env.get_local_goal())   # get updated local goal based on changed agent state
            speed_next = np.asarray(env.get_self_speed())  # ???
            #print('env.index:',env.index, goal_next, speed_next)
            state_next = [obs_stack, goal_next, speed_next]    # original state declare like 'state = [obs_stack, goal, speed]'
                                                               # get (updated l-r+) obs_stack, new local goal and updated speed_next
            # 4.1 get next state(pose)
            pose_ori_next = env.get_self_stateGT()   # 211019
            pose_next = np.asarray(pose_ori_next[:2])   # 211019

            if global_step % HORIZON == 0:   # every 128, estimate future V???
                state_next_list = comm.gather(state_next, root=0)
                pose_next_list = comm.gather(pose_next, root=0)   # 211027, for future usage
                '''
                if env.index == 0:
                    if flow_map is None:  # create flowmap
                        #flow_map = [[0,1,1,0,0,0,1,1,0,0],[1,0,1,0,0,1,0,0],[2,3,2,0,0,2,0],...,[2,4,2,0,0,2,0,2,0]]   # like list. # = count of how many human detected?
                        flow_map = get_flowmap(robot_curr_pose,humans_pose,update=false)
                        flow_map = np.array(flow_map)
                    else:                 # update flowmap
                        flow_map = get_flowmap(robot_curr_pose,humans_pose,update=true)
                        # flow_map : 12*12 dig0 matrix. axis=1 means the weights? or direction
                '''

                #last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy, action_bound=action_bound)
                last_v, _, _, _ = generate_action_human(env=env, state_list=state_list, pose_list=pose_list, policy=policy, action_bound=action_bound)   # from orca, 211020
                '''
                generate_action_rvo_dense(...,+flow_map)
                '''
                
            # 5. add transitons in buff and update policy
            r_list = comm.gather(r, root=0)   # e.g. r_list =  [-0.23433709215698428, 0.24986903841139441, 0.18645341029055684, 0.0, 0.016304648273046674])  # index=5
            terminal_list = comm.gather(terminal, root=0)   # e.g. [F, F, F, F, F] ... most... [F, F, F, T, F]
            #print('terminal list:',terminal_list)

            if env.index == 0:  # maybe env.index=0 means robot? or just one(VIP) act as?
                buff.append((state_list, a, r_list, terminal_list, logprob, v))   # intial buff = []  cummulnatively stacking buff for 128
                # 3 stacked lidar+relative dist+vel, [[1.23,232],...,[1.123,2.323] #5], [0.212, ... 3 ..., 0.112], [F, F, F, F, F], [-2.232, ..., 02.222], [-0.222, ..., -0.222]
                #print(env.index,'mmm,',buff)
                if len(buff) > HORIZON - 1:   # buffer exceed 128   # this part is for PPO
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)   # from model.ppo, batched buffer
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,  # r_batch, 0.99, v_batch
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)   # last_v(every 128, future v), terminal list, 0.95
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    # TODO Real training part
                    ppo_update_city(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,  # CNNPolicy, Adam, 1024, above lie about memory
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,  # 2, 5e-4, 0.1, 128
                                            num_env=NUM_ENV, frames=LASER_HIST,   # 20, 3
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)   # 512, 2
                    #print('policy:',policy, 'opt:',optimizer, 'memory:',memory, )

                    # 211026. for tensorboardX, network debug
                    #act_fea_cv1params = policy.state_dict()['act_fea_cv1.weight']
                    #actor1params =  policy.state_dict()['actor1.weight']
                    #queryparams = policy.state_dict()['query.bias'][0]
                    #keyparams = policy.state_dict()['key.bias'][0]
                    #valueparams = policy.state_dict()['value.bias'][0]
                    #act_fc3params =  policy.state_dict()['act_fc3.bias']
                    #writer.add_scalar('query.bias',queryparams,global_step=global_update)
                    #writer.add_scalar('key.bias', keyparams, global_step=global_update)
                    #writer.add_scalar('value.bias', valueparams, global_step=global_update)
                    #writer.add_scalar('act_fea_cv1.weight', act_fea_cv1params[0][0][0], global_step=global_update)
                    #writer.add_scalar('actor1.weight', actor1params[0][0], global_step=global_update)
                    #writer.add_scalar('act_fc3.weight', act_fc3params[0], global_step=global_update)


                    buff = []    # clean buffer
                    global_update += 1   # counting how many buffer transition and cleaned(how many time model updated)

            step += 1   # time goes on +1
            state = state_next
            
            pose = pose_next   # 2l.,j,j,11020
            #print(env.goal_point[0], env.init_pose[0], env.goal_point[1], env.init_pose[1])

        # after terminate = True(end step)
        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage_city_dense_{}'.format(global_update))   # save pth at every 20th model updated
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        #distance=2   # 211027 revive

        #print('envgoal:',env.goal_point)
        
        # 211027 only show and save env.index=0(robot) log
        if env.index ==0:
            logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, stepp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                        (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
            logger_cal.info(ep_reward)

        # 211026, for tensorboardX
        if env.index ==0:
            writer.add_scalar('episode reward of robot 0', ep_reward,global_step=global_update)

            info_p_lossss, info_v_lossss, info_entropyss = get_parameters()
            #info_p_lossss, info_v_lossss, info_entropyss, total_lossss = get_parameters()
            print(info_p_lossss,info_v_lossss,info_entropyss)
            writer.add_scalar('Policy(actor) Loss, vibrate, less then 1', info_p_lossss,global_step=global_update)
            writer.add_scalar('Value Loss, ???', info_v_lossss,global_step=global_update)
            writer.add_scalar('Entropy: How stochatic decisions of brain, decrease steady', info_entropyss,global_step=global_update)
            #writer.add_scalar('Total Loss', total_lossss,global_step=global_update)
            


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
    #print('env:',env.index)    # create rank'th different env
    reward = None
    action_bound = [[0, -1], [1, 1]]

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:   # (env.index=0))
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)    # 512, 2   # from model/net.py
        #policy = CNNPolicy(frames=LASER_HIST, action_space=2)   # 3, 2   # TODO: callback to this funct
        policy = RVOPolicy(frames=LASER_HIST, action_space=2)   # 3, 2
        
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):   # 'policy'
            os.makedirs(policy_path)

        #file = policy_path + '/stage_city_dense_340.pth'   # policy/stage3_2.pth
        file = policy_path + '/Stage_city_dense_160.pth'   # policy/stage3_2.pth
        #file = policy_path + '/Stage3_300.pth'   # policy/stage3_2.pth
        #print('file nave:',file)
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)   # comm, env.stageworld, 'policy', [[0, -1], [1, 1]], adam
    except KeyboardInterrupt:
        pass


writer.close()