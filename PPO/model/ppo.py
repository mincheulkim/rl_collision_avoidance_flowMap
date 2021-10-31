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


def generate_action(env, state_list, policy, action_bound):
    #print(state_list)
    #print('hey!!')
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
        #scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
        scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
        #print('a:',a, 'scaled_action:',scaled_action, 'a[0]:',a[0],'scaled_action_nobar:',np.clip(a, a_min=action_bound[0], a_max=action_bound[1]))
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

def generate_action_LM(env, state_list, pose_list, velocity_list, policy, action_bound):   # 211130
    
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        s_total_list, goal_total_list, speed_total_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        pose_list = np.asarray(pose_list)
        
        
        #local_map = []
        local_map = np.zeros((6,6))
        local_map_size = 3
        for i, pose in enumerate(pose_list):
            diff = pose-pose_list[0]
            #print(diff,': ', pose-pose_list[0])   # calculate diff position vs robot[0]
            mod_diff_x = np.floor(diff[0] / 1 + 6/2)
            mod_diff_y = np.ceil(diff[1] /1 - 6/2)
            mod_diff_y = np.abs(mod_diff_y)
            #print(i,' : ',mod_diff_x,mod_diff_y)
            if mod_diff_x >=0 and mod_diff_x <6 and mod_diff_y >=0 and mod_diff_y <6 and i is not 0:
                #print(i,' : ',np.int(mod_diff_x),np.int(mod_diff_y))
                local_map[np.int(mod_diff_y)][np.int(mod_diff_x)]=i
        

        speed_list_human, pose_list_human = [], []  # n-1
        
        for i in velocity_list:  # total # number of human's state
            speed_list_human.append(i)    # veloclity
        for i in pose_list:  # total # number of human's state
            pose_list_human.append(i)    # veloclity
        speed_list_human = np.asarray(speed_list_human)
        pose_list_human = np.asarray(pose_list_human)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

        # Build occupancy maps
        #occupancy_maps = build_occupancy_maps(human_states=pose_list_human, human_velocities=speed_list_human)   # just for one robot
        #local_maps = build_simple_LM(tot_pose=pose_list_human, tot_vel=speed_list_human)
        #print(occupancy_maps)


        v, a, logprob, mean = policy(s_list, goal_list, speed_list)
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action, local_map

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

def generate_action_human(env, state_list, pose_list, goal_global_list, num_env):   # pose_list added
    if env.index == 0:
        
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])      # lidar state
            goal_list.append(i[1])   # local goal
            speed_list.append(i[2])  # veloclity

        
        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        goal_list_new = goal_list  # 211021   nparray style, not torch as below
        speed_list = np.asarray(speed_list)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

        # 211020 pose_list create        
        p_list = []
        for i in pose_list:
            p_list.append(i)
        p_list = np.asarray(p_list)
        
        # Get action for humans(RVO)
        #sim = rvo2.PyRVOSimulator(1/60., num_env, 5, 3, 3, 0.4, 1)
        
        sim = rvo2.PyRVOSimulator(1/60., 3, 5, 5, 5, 0.5, 1)  # 211108   # neighborDist, maxNeighbors, timeHorizon, TimeHorizonObst, radius, maxspeed
        #callback

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
def generate_action_human_groups(env, state_list, pose_list, goal_global_list, num_env):
    if env.index == 0:

        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])      # lidar state
            goal_list.append(i[1])   # local goal
            speed_list.append(i[2])  # veloclity

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        goal_list_new = goal_list  # 211021   nparray style, not torch as below
        speed_list = np.asarray(speed_list)

        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

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

    #print('update')


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

def build_simple_LM(tot_pose, tot_vel):  # velocity: nonholonomic_robot, human_velocities=holonomic robot
    '''
    param human_states:
    return: tensor of shape
    '''
    

    #other_humans_pose = np.concatenate([np.array([(other_human.px, other_human.py)]) for other_human in human_states[1:]], axis=0)   # except robot self
    #other_humans_vel = np.concatenate([np.array([(other_human.vx, other_human.vy)]) for other_human in human_velocities[1:]], axis=0)   # except robot self
    other_humans_pose = tot_pose[1:]
    other_humans_vel = tot_vel[1:]
    other_px = other_humans_pose[:,0] - tot_pose[0][0]
    other_py = other_humans_pose[:,1] - tot_pose[0][1]
    print('1st:',other_px, other_py)
    # new x-axis is in the direction of robot's velocity
    
        #print('2nd:',other_px, other_py)

    # compute indicies of humans in the grid
    cell_size = 1
    cell_num = 6
    om_channel_size = 1

    occupancy_maps = np.zeros((6, 6))
    other_x_index = np.floor(other_px / cell_size + cell_num / 2)   # other_px / 1 + 2
    other_y_index = np.ceil(other_py / cell_size - cell_num / 2)
    other_y_index = np.abs(other_y_index)
        #print('other_x_index=',other_x_index,other_y_index)
    print('ox:',other_x_index,'oy:',other_y_index)
    other_x_index[other_x_index < 0] = 999
    other_x_index[other_x_index >= cell_num] = 999
    other_y_index[other_y_index < 0] = 999
    other_y_index[other_y_index >= cell_num] = 999
    print('refined index=',other_x_index, other_y_index)

    #OM shape: center is robot
    if om_channel_size == 1:   # just only consider position data
        print('damm')
        
        
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