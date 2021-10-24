import os #test
import logging
import sys
import socket
import numpy as np
from torch.optim.optimizer import Optimizer #test
import rospy #test
import torch #test
import torch.nn as nn #test
from mpi4py import MPI #test

from torch.optim import Adam #test
from collections import deque #test

from model.net import MLPPolicy, CNNPolicy #test
from stage_world1 import StageWorld #test
from model.ppo import ppo_update_stage1, generate_train_data
from model.ppo import generate_action, generate_action_human
from model.ppo import transform_buffer #test
from itertools import islice


MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
#HORIZON = 1000    # originaly 128. maybe it is memory size?
HORIZON = 3000    # for city scene
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5


#policy_path, optimizer 

def run(comm, env, policy, policy_path, action_bound, optimizer):
    # rate = rospy.Rate(5)
    buff = []
    global_update = 0
    global_step = 0


    if env.index == 0:
        env.reset_world()

    for id in range(MAX_EPISODES):
        #test
        #env.reset_pose()
        #env.generate_goal_point()

        # use this one!
        env.generate_pose_goal_circle()  # shafeshift above two line


        terminal = False
        ep_reward = 0
        step = 1

        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        speed_poly = np.asarray(env.get_self_speed_poly())  # 211103
        pose_ori = env.get_self_stateGT()   # 211019
        pose = np.asarray(pose_ori[:2])   # 211019
        rot = np.asarray(pose_ori[2])

        goal_global = np.asarray(env.get_goal_point())

        while not terminal and not rospy.is_shutdown():
        
            state_list = comm.gather(state, root=0)
            pose_list = comm.gather(pose, root=0)     # 211019. 5 states for each human
            goal_global_list = comm.gather(goal_global, root=0)
            
            # TODO add humans action

            # generate humans action_space
            
            #human_actions=generate_action_human(env=env, state_list=state_list, pose_list=pose_list, goal_global_list=goal_global_list, num_env=NUM_ENV)   # from orca, 21102
            human_actions=generate_action_human(env=env, state_list=state_list, pose_list=pose_list, goal_global_list=goal_global_list, num_env=NUM_ENV)   # from orca, 21102
            #print('human actions:',human_actions)

            #print('state_list:',len(state_list))
            #print('state=',len(state))
            
            #print(np.array(state_list).shape,np.array(state).shape)

            
            # generate robot action (at rank==0)
            if env.index == 0:
                v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # execute actions
            real_action = comm.scatter(human_actions, root=0)

            # distribute actions btwn robot and humans
            if env.index == 0:
                env.control_vel(scaled_action)
            else: # pre-RVO vel, humans
                angles = np.arctan2(real_action[1], real_action[0])
                diff = angles - rot
                length = np.sqrt([real_action[0]**2+real_action[1]**2])
                mod_vel = (length, diff)
                env.control_vel(mod_vel)   # 211108

            # rate.sleep()
            rospy.sleep(0.001)

            # TODO check collision via sanity check
            # get min(lidar) < threshold => collision

            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
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
            #print("len(state_next)")
            #print(len(state_next))
            ############training#######################################################################################

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                #last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy, action_bound=action_bound)
                if env.index ==0:
                    last_v, _, _, _ = generate_action(env=env, state_list=state_list, policy=policy, action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)

            if env.index == 0:

                #TODO. if local_map: change buff.append
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                print(a, r_list, terminal_list, logprob, v)
                # (array([[ 2.4515967, -1.1015964]], dtype=float32), [0.0], [False], array([[-4.3833094]], dtype=float32), array([[-0.07423574]], dtype=float32))
                #print(state_list)
                # deque(array[],array[],array[]), array[1,2], array[1,2])
                #buff.append((state_list, a, r, terminal, logprob, v))

                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            #num_env=NUM_ENV, frames=LASER_HIST,
                                            num_env=1, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1
                    print('update ppo:',global_update,' th global steps')

            step += 1

            ###################################################################################################
            state = state_next
            pose = pose_next   # 2l.,j,j,11020
            speed_poly = speed_next_poly  # 211104
            rot = rot_next

        
        #####save policy and logger##############################################################################################
    
        if env.index == 0:
            if global_update != 0 and global_update % 10 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                torch.save(policy, policy_path + '/Stage1_{}_tot'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
            #distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
            #distance = 0

            logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Result %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, result))
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    
    print("RANK:",rank," ENV")
    
    reward = None
    action_bound = [[0, -1], [1, 1]] ####
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()
        

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        # Load total model

        #file = policy_path + '/Stage1_100'
        file = policy_path + '/_____'
        #file_tot = policy_path + '/stage_____tot'
        file_tot = policy_path + '/Stage1_10_tot'
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
