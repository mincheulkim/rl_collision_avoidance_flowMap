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


def transform_buffer(buff):    # from 5 step at ppo_stage3.py  
    # buff=3 stacked lidar+relative dist+vel, [[1.23,232],...,[1.123,2.323] #5], [0.212, ... 3 ..., 0.112], [F, F, F, F, F], [-2.232, ..., 02.222], [-0.222, ..., -0.222]
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff:
        for state in e[0]:   # states data
            s_temp.append(state[0])   # 1. lidar
            goal_temp.append(state[1])   # 2. local goal
            speed_temp.append(state[2])   # 3. velocity
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])   # A
        r_batch.append(e[2])   # reward_list
        d_batch.append(e[3])   # terminal list(T or F)
        l_batch.append(e[4])   # logprob
        v_batch.append(e[5])   # 

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
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

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

def generate_action_rvo(env, state_list, pose_list, policy, action_bound):   # pose_list added
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
        
        #p_list = Variable(torch.from_numpy(p_list)).float().cuda()  # to tensor
        #print('goal_list_new', tuple(goal_list_new[0]))
        #print('p_list:',tuple(p_list[0]))

        #slice tuple(x,y) -> (x), (y)
        #print('ori_goal_list_new', tuple(goal_list_new[0]))    # original (x, y)
        #print('x_goal_list_new', tuple(goal_list_new[0])[:1])  # slice (x,)
        #print('y_goal_list_new', tuple(goal_list_new[0])[1:2]) # slice (y,)
        
        #print('changed:',tuple(goal_list[0]))   # nd_array to tuple   https://www.kite.com/python/answers/how-to-convert-a-numpy-array-to-a-tuple-in-python
        #print('changed:',tuple(p_list[0]))   # nd_array to tuple   https://www.kite.com/python/answers/how-to-convert-a-numpy-array-to-a-tuple-in-python

        # TODO get action based rvo
        v, a, logprob, mean = policy(s_list, goal_list, speed_list, p_list)     # now create action from rvo(net.py.forward())
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        
        sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)
        #sim = rvo2.PyRVOSimulator(1/60., 10, 10, 5, 5, 0.3, 1)

        a0=sim.addAgent(tuple(p_list[0]))
        a1=sim.addAgent(tuple(p_list[1]))
        a2=sim.addAgent(tuple(p_list[2]))
        a3=sim.addAgent(tuple(p_list[3]))
        a4=sim.addAgent(tuple(p_list[4]))

        #print('p_list[0]=',p_list[0])
        #print('g_list[0]=',goal_list_new[0])
        #print('aaaa:',aaaa)

        h0v = goal_list_new[0]-p_list[0]   # velocity
        h1v = goal_list_new[1]-p_list[1]
        h2v = goal_list_new[2]-p_list[2]
        h3v = goal_list_new[3]-p_list[3]
        h4v = goal_list_new[4]-p_list[4]
        h0s = np.linalg.norm(h0v)          # speed
        h1s = np.linalg.norm(h1v)
        h2s = np.linalg.norm(h2v)
        h3s = np.linalg.norm(h3v)
        h4s = np.linalg.norm(h4v)
        prefv0=h0v/h0s if h0s >1 else h0v
        prefv1=h1v/h1s if h1s >1 else h1v
        prefv2=h2v/h2s if h2s >1 else h2v
        prefv3=h3v/h3s if h3s >1 else h3v
        prefv4=h4v/h4s if h4s >1 else h4v

        sim.setAgentPrefVelocity(a0, tuple(prefv0))
        sim.setAgentPrefVelocity(a1, tuple(prefv1))
        sim.setAgentPrefVelocity(a2, tuple(prefv2))
        sim.setAgentPrefVelocity(a3, tuple(prefv3))
        sim.setAgentPrefVelocity(a4, tuple(prefv4))

        sim.doStep()

        velocitiys = [sim.getAgentVelocity(0), sim.getAgentVelocity(1), sim.getAgentVelocity(2), sim.getAgentVelocity(3), sim.getAgentVelocity(4)]

        #print('orca vel:',velocitiys)

        

        #print(prefv0,prefv1,prefv2,prefv3,prefv4)
        
        

        
        #v1 = (tuple(goal_list_new[0])[:1]-tuple(p_list[0])[:1], tuple(goal_list_new[0])[1:2]-tuple(p_list[0])[1:2])
        #print('v1:',v1)
        #      gx-px, gy-py

        #v1 = (tuple(goal_list[0][0])-tuple(p_list[0][0]),tuple(goal_list[0][1])-tuple(p_list[0][1]))
        #print('v1:',v1)

        #scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
        #scaled_action = np.array([[0.123,0.00],[0.23,0.00],[0.33,0.00],[0.43,0.00],[0.232,0.003]])
        scaled_action = sim.getAgentVelocity(0), sim.getAgentVelocity(1), sim.getAgentVelocity(2), sim.getAgentVelocity(3), sim.getAgentVelocity(4)
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action



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
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

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

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)   # model/net.py  into the network inputs
                                                  #policy = CNN policy(input lidar data, goal, velocity)

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

    print('update')


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

    obss = np.delete(obss, filter_index, 0)
    goals = np.delete(goals, filter_index, 0)
    speeds = np.delete(speeds, filter_index, 0)
    actions = np.delete(actions, filter_index, 0)
    logprobs  = np.delete(logprobs, filter_index, 0)
    advs = np.delete(advs, filter_index, 0)
    targets = np.delete(targets, filter_index, 0)


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

def ppo_update_stage3(policy, optimizer, batch_size, memory, epoch,   # # CNNPolicy, Adam, 1024, above lie about memory, epoch=2
               coeff_entropy=0.02, clip_value=0.2,    #  coeff_entropy= 5e-4, clip_val = 0.1
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):  # num_step= 128, num_env=5, frames(laser_hist)=3, obs_size=512, act_size=2
    # num_env=12 is default, ppo_stage2.py line 33 is real
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory   # (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

    advs = (advs - advs.mean()) / advs.std()   # Advantage normalize?

    # 128= batch training data, num_env = agent num
    obss = obss.reshape((num_step*num_env, frames, obs_size))   # 128*5, 3, 512
    goals = goals.reshape((num_step*num_env, 2))   # 128*5, 2(x,y)
    speeds = speeds.reshape((num_step*num_env, 2))  # 128*5, 2(vx,vy)
    actions = actions.reshape(num_step*num_env, act_size)  # 128*5, 2
    logprobs = logprobs.reshape(num_step*num_env, 1)  # 128*5, 1(logprob e.g. -2.12323)
    advs = advs.reshape(num_step*num_env, 1)  # same
    targets = targets.reshape(num_step*num_env, 1)  # targets?

    for update in range(epoch):  # 0, 1, 2
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))

    print('update')

def ppo_update_city(policy, optimizer, batch_size, memory, epoch,   # # CNNPolicy, Adam, 1024, above lie about memory, epoch=2
               coeff_entropy=0.02, clip_value=0.2,    #  coeff_entropy= 5e-4, clip_val = 0.1
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):  # num_step= 128, num_env=5, frames(laser_hist)=3, obs_size=512, act_size=2
    # num_env=12 is default, ppo_stage2.py line 33 is real
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory   # (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

    advs = (advs - advs.mean()) / advs.std()   # Advantage normalize?

    # 128= batch training data, num_env = agent num
    obss = obss.reshape((num_step*num_env, frames, obs_size))   # 128*5, 3, 512
    goals = goals.reshape((num_step*num_env, 2))   # 128*5, 2(x,y)
    speeds = speeds.reshape((num_step*num_env, 2))  # 128*5, 2(vx,vy)
    actions = actions.reshape(num_step*num_env, act_size)  # 128*5, 2
    logprobs = logprobs.reshape(num_step*num_env, 1)  # 128*5, 1(logprob e.g. -2.12323)
    advs = advs.reshape(num_step*num_env, 1)  # same
    targets = targets.reshape(num_step*num_env, 1)  # targets?

    for update in range(epoch):  # 0, 1, 2
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))

    print('update')


