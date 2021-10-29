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


MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
#HORIZON = 1024    # originaly 128. maybe it is memory size?. for static obstacle scene?
#HORIZON = 3000    # for city scene
#HORIZON = 2048    # v3
HORIZON = 3072    # v4 static obstacle
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024   # is small batch is good? 64?
#BATCH_SIZE = 128   # is small batch is good? 64?
EPOCH = 2
COEFF_ENTROPY = 5e-4
#CLIP_VALUE = 0.1
CLIP_VALUE = 0.2
NUM_ENV = 6
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5


#policy_path, optimizer 

def run(comm, env, policy, policy_path, action_bound, optimizer):
    # rate = rospy.Rate(5)
    buff = []
    buff_c = []
    global_update = 0
    global_step = 0

    memory_size = 0


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

            env_index_list = comm.gather(env.index, root=0)    # 0,1,2,3,4,5
            
            if env.index==0:
                state_list_new = state_list[0:1]   # 211126 https://jinisbonusbook.tistory.com/32
            
            #print('envindex=',env_index_list)
            
            # TODO add humans action

            # generate humans action_space
            
            human_actions=generate_action_human(env=env, state_list=state_list, pose_list=pose_list, goal_global_list=goal_global_list, num_env=NUM_ENV)   # from orca, 21102
            # erase me!
            #human_actions=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
            #print('human actions:',human_actions)

            # generate robot action (at rank==0)
            # ERASE ME!
            if env.index==0:

                #v, a, logprob, scaled_action=generate_action(env=env, state_list=state_robot, policy=policy, action_bound=action_bound)
                #v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list, policy=policy, action_bound=action_bound)
                v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list_new, policy=policy, action_bound=action_bound)

                #print('diff_2:',state_list[0][1],state_robot[0][1])                
                #print('diff_3:',state_list[0][2],state_robot[0][2])                
                #v_b, a_b, logprob_b, scaled_action_b=generate_action(env=env, state_list=state_list, policy=policy, action_bound=action_bound)
                #v_c, a_c, logprob_c, scaled_action_c=generate_action(env=env, state_list=state_robot, policy=policy, action_bound=action_bound)
                # For debugging
                #print('#####round 1#####')
                #print('original v,a,logprob,scaled_action:',v,a,logprob,scaled_action)
                #print('original(double) v,a,logprob,scaled_action:',v_b,a_b,logprob_b,scaled_action_b)
                #print('modified v,a,logprob,scaled_action:',v_c, a_c, logprob_c, scaled_action_c)
                #print('original state:',state_list)
                #print('modified state:',state_robot)

            # execute actions
            real_action = comm.scatter(human_actions, root=0)

            # distribute actions btwn robot and humans
            if env.index == 0:
                env.control_vel(scaled_action)
            else: # pre-RVO vel, humans
                angles = np.arctan2(real_action[1], real_action[0])
                diff = angles - rot
                length = np.sqrt([real_action[0]**2+real_action[1]**2])
                #mod_vel = (length, diff)
                mod_vel = (length/2, diff/2)   # make more ease, erase me!
                # Erase me!!
                env.control_vel(mod_vel)   # 211108
                #env.control_vel([0,0])   # 211108
                

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
                if env.index == 0:
                    state_next_list_new = state_next_list[0:1]
                if env.index ==0:

                    #last_v_r, _, _, _ = generate_action(env=env, state_list=state_next_robot, policy=policy, action_bound=action_bound)
                    #last_v_r, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy, action_bound=action_bound)
                    last_v_r, _, _, _ = generate_action(env=env, state_list=state_next_list_new, policy=policy, action_bound=action_bound)
                    
                    #last_v_c, _, _, _ = generate_action(env=env, state_list=state_next_robot, policy=policy, action_bound=action_bound)
                    #print('####round 2#####')
                    #print('original last_v:',last_v)
                    #print('modified last_v:',last_v_c)
                else:
                    last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy, action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)
            if env.index ==0:
                r_list_new = r_list[0:1]
                terminal_list_new=terminal_list[0:1]

            #if env.index == 0:
            if env.index == 0 and not (step == 1 and terminal):

                #TODO. if local_map: change buff.append
                
                
                #buff.append((state_list, a, r_list, terminal_list, logprob, v))   # state_list, r_list, terminal_list: can we manage
                #buff.append((state_robot, a, r_robot, terminal_robot, logprob, v))   # state_list, r_list, terminal_list: can we manage
                #print(len(state_list_new), len(a), len(r_list_new), len(terminal_list_new), len(logprob), len(v))
                buff.append((state_list_new, a, r_list_new, terminal_list_new, logprob, v))   # new


                # TODO check a, r_robot(real plus root), terminal_robot
                #print('original reward:',r_list,'original_terminal:',terminal_list,'mod reward:',r_robot,'mod terminal:',terminal_robot,'ep_reward:',ep_reward)
                memory_size += 1

                #buff_c.append((state_robot, a_c, r_robot, terminal_robot, logprob_c, v_c))   # state_list, r_list, terminal_list: can we manage
                #print('#####Round 3######')
                #print('original buff:',buff)
                #print('modified buff:',buff_c)
                
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v_r, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            #num_env=NUM_ENV, frames=LASER_HIST,
                                            num_env=1, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1
                    

            step += 1
            #print('memory size:',memory_size,'len buffer:',len(buff))
            ###################################################################################################
            state = state_next
            pose = pose_next   # 2l.,j,j,11020
            speed_poly = speed_next_poly  # 211104
            rot = rot_next


        
        #####save policy and logger##############################################################################################
    
        if env.index == 0:
            if global_update != 0 and global_update % 5 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                torch.save(policy, policy_path + '/Stage1_{}_tot'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
            #distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
            #distance = 0
            if not (step==2 and terminal):
                logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Result %s, MemSize: %05d' % \
                        (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, result, memory_size))
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

        file = policy_path + '/Stage1_610'
        #file = policy_path + '/_____'
        file_tot = policy_path + '/stage_____tot'
        #file_tot = policy_path + '/Stage1_5_tot'
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
