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

def transform_buffer_r(buff_r, LM):    # 211101, for robot
    # buff=3 stacked lidar+relative dist+vel, [[1.23,232],...,[1.123,2.323] #5], [0.212, ... 3 ..., 0.112], [F, F, F, F, F], [-2.232, ..., 02.222], [-0.222, ..., -0.222]
    # buff = state, a_r, r, terminal, logprob_r, v_r
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch, occupancy_maps_batch = [], [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff_r:
        '''
        for state in e[0]:   # states data
            s_temp.append(state[0])   # 1. lidar
            goal_temp.append(state[1])   # 2. local goal
            speed_temp.append(state[2])   # 3. velocity
        '''
        s_temp.append(e[0][0])   # 1. lidar
        goal_temp.append(e[0][1])   # 2. local goal
        speed_temp.append(e[0][2])   # 3. velocity
            
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)

        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])   # A
        r_batch.append(e[2])   # reward
        d_batch.append(e[3])   # terminal(T or F)
        l_batch.append(e[4])   # logprob
        v_batch.append(e[5])   # V
        if LM:
            occupancy_maps_batch.append(e[6])   # Occupancy map

    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)
    if LM:
        occupancy_maps_batch = np.asarray(occupancy_maps_batch)

    if LM:
        return s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch, occupancy_maps_batch
    else:
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
        
        #sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)
        sim = rvo2.PyRVOSimulator(1/60., 1, 5, 1.5, 1.5, 0.4, 1)
        #sim = rvo2.PyRVOSimulator(1/60., 10, 10, 5, 5, 0.3, 1)

        a0=sim.addAgent(tuple(p_list[0]))
        a1=sim.addAgent(tuple(p_list[1]))
        a2=sim.addAgent(tuple(p_list[2]))
        a3=sim.addAgent(tuple(p_list[3]))
        a4=sim.addAgent(tuple(p_list[4]))

        #print('p_list[0]=',p_list[0])
        #print('g_list[0]=',goal_list_new[0])
        #print('aaaa:',aaaa)

        '''
        h0v = goal_list_new[0]-p_list[0]   # velocity
        h1v = goal_list_new[1]-p_list[1]
        h2v = goal_list_new[2]-p_list[2]
        h3v = goal_list_new[3]-p_list[3]
        h4v = goal_list_new[4]-p_list[4]
        '''

        h0v = goal_list_new[0]   # TODO because goal's here is local goal, there is no need to minus current position!!!
        h1v = goal_list_new[1]
        h2v = goal_list_new[2]
        h3v = goal_list_new[3]
        h4v = goal_list_new[4]

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

        #scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
        #scaled_action = np.array([[0.123,0.00],[0.23,0.00],[0.33,0.00],[0.43,0.00],[0.232,0.003]])
        scaled_action = sim.getAgentVelocity(0), sim.getAgentVelocity(1), sim.getAgentVelocity(2), sim.getAgentVelocity(3), sim.getAgentVelocity(4)
        #scaled_action = sim.getAgentVelocity(0), sim.getAgentVelocity(1), sim.getAgentVelocity(2), (0.5,0.0), (0.5,0.0)
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

def generate_action_rvo_dense(env, state_list, pose_list, policy, action_bound):   # pose_list added
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        #s_list, goal_list, speed_list, flowmap = [], [], [], []
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

        # get action based rvo
        v, a, logprob, mean = policy(s_list, goal_list, speed_list, p_list)     # now create action from rvo(net.py.forward())
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        # TODO for robot
        #v, a, logprob, mean = policy(s_list, goal_list, speed_list, p_list)     # now create action from rvo(net.py.forward())
        
        #sim = rvo2.PyRVOSimulator(1/60., 1, 5, 1.5, 1.5, 0.4, 1)
        sim = rvo2.PyRVOSimulator(1/60., 3, 5, 5, 5, 0.4, 1)

        a0=sim.addAgent(tuple(p_list[0]))
        a1=sim.addAgent(tuple(p_list[1]))
        a2=sim.addAgent(tuple(p_list[2]))
        a3=sim.addAgent(tuple(p_list[3]))
        a4=sim.addAgent(tuple(p_list[4]))
        a5=sim.addAgent(tuple(p_list[5]))
        a6=sim.addAgent(tuple(p_list[6]))
        a7=sim.addAgent(tuple(p_list[7]))
        a8=sim.addAgent(tuple(p_list[8]))
        a9=sim.addAgent(tuple(p_list[9]))
        a10=sim.addAgent(tuple(p_list[10]))
        a11=sim.addAgent(tuple(p_list[11]))
        a12=sim.addAgent(tuple(p_list[12]))
        a13=sim.addAgent(tuple(p_list[13]))
        a14=sim.addAgent(tuple(p_list[14]))
        a15=sim.addAgent(tuple(p_list[15]))
        a16=sim.addAgent(tuple(p_list[16]))
        a17=sim.addAgent(tuple(p_list[17]))
        a18=sim.addAgent(tuple(p_list[18]))
        a19=sim.addAgent(tuple(p_list[19]))

        # Obstacles are also supported # 211022   https://gamma.cs.unc.edu/RVO2/documentation/2.0/class_r_v_o_1_1_r_v_o_simulator.html#a0f4a896c78fc09240083faf2962a69f2
        #o1 = sim.addObstacle([(2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)])
        #sim.processObstacles()
        # TODO concern about local obstacle

        h0v = goal_list_new[0]   # TODO because goal's here is local goal, there is no need to minus current position
        h1v = goal_list_new[1]
        h2v = goal_list_new[2]
        h3v = goal_list_new[3]
        h4v = goal_list_new[4]
        h5v = goal_list_new[5]  
        h6v = goal_list_new[6]
        h7v = goal_list_new[7]
        h8v = goal_list_new[8]
        h9v = goal_list_new[9]
        h10v = goal_list_new[10] 
        h11v = goal_list_new[11]
        h12v = goal_list_new[12]
        h13v = goal_list_new[13]
        h14v = goal_list_new[14]
        h15v = goal_list_new[15]  
        h16v = goal_list_new[16]
        h17v = goal_list_new[17]
        h18v = goal_list_new[18]
        h19v = goal_list_new[19]

        h0s = np.linalg.norm(h0v)          # speed
        h1s = np.linalg.norm(h1v)
        h2s = np.linalg.norm(h2v)
        h3s = np.linalg.norm(h3v)
        h4s = np.linalg.norm(h4v)
        h5s = np.linalg.norm(h5v)          # speed
        h6s = np.linalg.norm(h6v)
        h7s = np.linalg.norm(h7v)
        h8s = np.linalg.norm(h8v)
        h9s = np.linalg.norm(h9v)
        h10s = np.linalg.norm(h10v)          # speed
        h11s = np.linalg.norm(h11v)
        h12s = np.linalg.norm(h12v)
        h13s = np.linalg.norm(h13v)
        h14s = np.linalg.norm(h14v)
        h15s = np.linalg.norm(h15v)          # speed
        h16s = np.linalg.norm(h16v)
        h17s = np.linalg.norm(h17v)
        h18s = np.linalg.norm(h18v)
        h19s = np.linalg.norm(h19v)

        prefv0=h0v/h0s if h0s >1 else h0v
        prefv1=h1v/h1s if h1s >1 else h1v
        prefv2=h2v/h2s if h2s >1 else h2v
        prefv3=h3v/h3s if h3s >1 else h3v
        prefv4=h4v/h4s if h4s >1 else h4v
        prefv5=h5v/h5s if h5s >1 else h5v
        prefv6=h6v/h6s if h6s >1 else h6v
        prefv7=h7v/h7s if h7s >1 else h7v
        prefv8=h8v/h8s if h8s >1 else h8v
        prefv9=h9v/h9s if h9s >1 else h9v
        prefv10=h10v/h10s if h10s >1 else h10v
        prefv11=h11v/h11s if h11s >1 else h11v
        prefv12=h12v/h12s if h12s >1 else h12v
        prefv13=h13v/h13s if h13s >1 else h13v
        prefv14=h14v/h14s if h14s >1 else h14v
        prefv15=h15v/h15s if h15s >1 else h15v
        prefv16=h16v/h16s if h16s >1 else h16v
        prefv17=h17v/h17s if h17s >1 else h17v
        prefv18=h18v/h18s if h18s >1 else h18v
        prefv19=h19v/h19s if h19s >1 else h19v

        sim.setAgentPrefVelocity(a0, tuple(prefv0))
        sim.setAgentPrefVelocity(a1, tuple(prefv1))
        sim.setAgentPrefVelocity(a2, tuple(prefv2))
        sim.setAgentPrefVelocity(a3, tuple(prefv3))
        sim.setAgentPrefVelocity(a4, tuple(prefv4))
        sim.setAgentPrefVelocity(a5, tuple(prefv5))
        sim.setAgentPrefVelocity(a6, tuple(prefv6))
        sim.setAgentPrefVelocity(a7, tuple(prefv7))
        sim.setAgentPrefVelocity(a8, tuple(prefv8))
        sim.setAgentPrefVelocity(a9, tuple(prefv9))
        sim.setAgentPrefVelocity(a10, tuple(prefv10))
        sim.setAgentPrefVelocity(a11, tuple(prefv11))
        sim.setAgentPrefVelocity(a12, tuple(prefv12))
        sim.setAgentPrefVelocity(a13, tuple(prefv13))
        sim.setAgentPrefVelocity(a14, tuple(prefv14))
        sim.setAgentPrefVelocity(a15, tuple(prefv15))
        sim.setAgentPrefVelocity(a16, tuple(prefv16))
        sim.setAgentPrefVelocity(a17, tuple(prefv17))
        sim.setAgentPrefVelocity(a18, tuple(prefv18))
        sim.setAgentPrefVelocity(a19, tuple(prefv19))

        sim.doStep()

        #scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
        #scaled_action = np.array([[0.123,0.00],[0.23,0.00],[0.33,0.00],[0.43,0.00],[0.232,0.003]])
        scaled_action = sim.getAgentVelocity(0), sim.getAgentVelocity(1), sim.getAgentVelocity(2), sim.getAgentVelocity(3), sim.getAgentVelocity(4),sim.getAgentVelocity(5), sim.getAgentVelocity(6), sim.getAgentVelocity(7), sim.getAgentVelocity(8), sim.getAgentVelocity(9), sim.getAgentVelocity(10), sim.getAgentVelocity(11), sim.getAgentVelocity(12), sim.getAgentVelocity(13), sim.getAgentVelocity(14),sim.getAgentVelocity(15), sim.getAgentVelocity(16), sim.getAgentVelocity(17), sim.getAgentVelocity(18), sim.getAgentVelocity(19)
        #scaled_action = sim.getAgentVelocity(0), sim.getAgentVelocity(1), sim.getAgentVelocity(2), (0.5,0.0), (0.5,0.0)
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

def generate_action_human(env, state_list, pose_list, action_bound, velocity_poly_list, goal_global_list, rot_list, num_env):   # pose_list added
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
                hv = goal_global_list[i] - p_list[i]  # TODO because goal's here is local goal, there is no need to minus current position
                hs = np.linalg.norm(hv)     # 211027   get raw_scaled action from learned policy
                prefv=hv/hs if hs >1 else hv
                sim.setAgentPrefVelocity(i-1, tuple(prefv))
                
        # Obstacles are also supported # 211022   https://gamma.cs.unc.edu/RVO2/documentation/2.0/class_r_v_o_1_1_r_v_o_simulator.html#a0f4a896c78fc09240083faf2962a69f2
        #o1 = sim.addObstacle([(2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)])
        #sim.processObstacles()
        # TODO concern about local obstacle

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

def generate_action_robot(env, state, pose, policy, action_bound, evaluate):   # policy = RobotPolicy
    if env.index == 0:
        s_list, goal_list, speed_list, p_list = [], [], [], []
        '''
        for i in state:
            s_list.append(i[0])      # lidar state
            goal_list.append(i[1])   # local goal
            speed_list.append(i[2])  # veloclity
        '''
        
        s_list = state[0]
        goal_list = state[1]
        speed_list = state[2]
        p_list = pose
             
        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)
        p_list = np.asarray(p_list)

        #s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        #goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        #speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
        s_list = Variable(torch.from_numpy(s_list).unsqueeze(dim=0)).float().cuda()   # (3, 512)   -> make (1, 3, 512)   # 1: num of agent(gather)
        goal_list = Variable(torch.from_numpy(goal_list).unsqueeze(dim=0)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list).unsqueeze(dim=0)).float().cuda()   # erase cuda()

        #print('policy:',policy)
        # Get action for robot(RobotPolicy)

        v, a, logprob, mean = policy(s_list, goal_list, speed_list, p_list)     # now create action from rvo(net.py.forward())
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        raw_scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])  # for Robot      # a[0] = (linear, angular), [0,-1], [1, 1]
                            # TODO. see is action_bound really work

        # for evaluate(best action)
        if evaluate:
            mean = mean.data.cpu().numpy()
            scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])   

        scaled_action = raw_scaled_action

        
        
    else:  # env.index =! 0
        v = None
        a = None
        scaled_action = None
        logprob = None
    
    return v, a, logprob, scaled_action

def generate_action_robot_localmap(env, state, pose, policy, action_bound, state_list, pose_list, velocity_poly_list, evaluate):   # 211001 for local mapping
    if env.index == 0:
        #print(velocity_poly_list)
        s_list, goal_list, speed_list, p_list = [], [], [], []
        
        s_list = state[0]
        goal_list = state[1]
        speed_list = state[2]
        p_list = pose

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)
        speed_ori = speed_list
        p_list = np.asarray(p_list)

        s_list = Variable(torch.from_numpy(s_list).unsqueeze(dim=0)).float().cuda()   # (3, 512)   -> make (1, 3, 512)   # 1: num of agent(gather)
        goal_list = Variable(torch.from_numpy(goal_list).unsqueeze(dim=0)).float().cuda()   # (1, 2)
        speed_list = Variable(torch.from_numpy(speed_list).unsqueeze(dim=0)).float().cuda()   # (1, 2))

        # TODO build occupancy map
        # get humans state
        speed_list_human, pose_list_human = [], []  # n-1
        
        for i in velocity_poly_list:  # total # number of human's state
            speed_list_human.append(i)    # veloclity
        for i in pose_list:  # total # number of human's state
            pose_list_human.append(i)    # veloclity
        speed_list_human = np.asarray(speed_list_human)
        pose_list_human = np.asarray(pose_list_human)
        '''
        print('state:',p_list)
        print('ve:', speed_list)
        print('speed:',speed_list_human)
        print('pose:', pose_list_human)
        '''    

        occupancy_maps = build_occupancy_maps(state=p_list, velocity=speed_ori, human_states=pose_list_human, human_velocities=speed_list_human)   # just for one robot
        #print(occupancy_maps)
        occupancy_maps_list = np.asarray(occupancy_maps)
        occupancy_maps_list = Variable(torch.from_numpy(occupancy_maps_list).squeeze(0)).float().cuda()   # (1,1,48) -> (1,48)
        
        #print(occupancy_maps_list, occupancy_maps_list.shape, s_list.shape, goal_list.shape, speed_list.shape)

        #v, a, logprob, mean = policy(s_list, goal_list, speed_list, p_list)
        v, a, logprob, mean = policy(s_list, goal_list, speed_list, occupancy_maps_list)
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        raw_scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])  # for Robot      # [0,-1], [1, 1]    # a = a[0], only 1 item

        if evaluate:
            mean = mean.data.cpu().numpy()
            scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
            
        scaled_action = raw_scaled_action
#        scaled_action = [1,0]
        
        
    else:  # env.index =! 0
        #print(state[2])   # linear vel, ang
        v = None
        a = None
        scaled_action = None
        logprob = None
        occupancy_maps = None

    return v, a, logprob, scaled_action, occupancy_maps



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

def generate_train_data_r(rewards, gamma, values, last_value, dones, lam):   # rewards=r_batch_r, gamma=0.99, values=v_batch_r, last_value=last_v_r, dones=d_batch_r, lam=LAMDA
    num_step = rewards.shape[0]
    #num_env = rewards.shape[1]
    num_env = 1
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        #delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]  # 211102 because indices error
        #gae = delta + gamma * lam * (1 - dones[t, :]) * gae
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae

        #targets[t, :] = gae + values[t, :]
        targets[t] = gae + values[t]

    #advs = targets - values[:-1, :]
    advs = targets - values[:-1]
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

    print('update_stage3')


info_p_losss = None
info_v_losss = None
info_entropys = None

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
    advs = advs.reshape(num_step*num_env, 1)  # 128*5, 1
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
            policy_loss = -torch.min(surrogate1, surrogate2).mean()   # same as action loss @ 211027

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)       # value loss @ 211027

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy   # 20 is value_loss_coefficient? maybe?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(dist_entropy.detach().cpu().numpy())
            # logger_ppo.info('p_loss: {}, v_loss: {}, entropy: {}'.format(info_p_loss, info_v_loss, info_entropy))
            
            # 211027 for logging     # https://www.infoking.site/64
            global info_p_losss
            info_p_losss = info_p_loss
            global info_v_losss
            info_v_losss = info_v_loss
            global info_entropys
            info_entropys = info_entropy
        
            

    print('update_city')

def ppo_update_city_r(policy, optimizer, batch_size, memory, epoch,   # # CNNPolicy, Adam, 1024, above lie about memory, epoch=2
               coeff_entropy=0.02, clip_value=0.2,    #  coeff_entropy= 5e-4, clip_val = 0.1
               num_step=2048, num_env=1, frames=1, obs_size=24, act_size=4, LM=False):  # num_step= 128, num_env=5, frames(laser_hist)=3, obs_size=512, act_size=2
    # num_env=12 is default, ppo_stage2.py line 33 is real
    #obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory   # (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
    if LM:
        obss, goals, speeds, actions, logprobs, targets, values, rewards, advs, occupancy_maps = memory   # (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
    else:
        obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory   # (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

    advs = (advs - advs.mean()) / advs.std()   # Advantage normalize?

    # 128= batch training data, num_env = agent num
    obss = obss.reshape((num_step*num_env, frames, obs_size))   # 128*1, 3, 512
    goals = goals.reshape((num_step*num_env, 2))   # 128*1, 2(x,y)
    speeds = speeds.reshape((num_step*num_env, 2))  # 128*1, 2(vx,vy)
    actions = actions.reshape(num_step*num_env, act_size)  # 128*1, 2
    if LM:
        occupancy_maps = occupancy_maps.reshape((num_step*num_env, 48))
    logprobs = logprobs.reshape(num_step*num_env, 1)  # 128*5, 1(logprob e.g. -2.12323)
    advs = advs.reshape(num_step*num_env, 1)  # 128*5, 1
    targets = targets.reshape(num_step*num_env, 1)  # targets?

    for update in range(epoch):  # 0, 1, 2
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()
            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            if LM:
                sampled_occupancy_maps = Variable(torch.from_numpy(occupancy_maps[index])).float().cuda()

            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()

            #new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)
            if LM:
                new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions, sampled_occupancy_maps)
            else:
                new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()   # same as action loss @ 211027

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)       # value loss @ 211027

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy   # 20 is value_loss_coefficient? maybe?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
            #                                         float(value_loss.detach().cpu().numpy()), float(dist_entropy.detach().cpu().numpy())
            #logger_ppo.info('p_loss_R: {}, v_loss_R: {}, entropy_R: {}'.format(info_p_loss, info_v_loss, info_entropy))
            # 211103 add total loss
            info_p_loss, info_v_loss, info_entropy, info_total_loss = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(dist_entropy.detach().cpu().numpy()), float(loss.detach().cpu().numpy())
            logger_ppo.info('p_loss_R: {}, v_loss_R: {}, entropy_R: {}, total_loss_R: {}'.format(info_p_loss, info_v_loss, info_entropy, info_total_loss))
            
            # 211027 for logging     # https://www.infoking.site/64
            global info_p_losss
            info_p_losss = info_p_loss
            global info_v_losss
            info_v_losss = info_v_loss
            global info_entropys
            info_entropys = info_entropy
        
    print('update_city_r')


#total_losss = None

def get_parameters():   # 211027 for logging
    #return info_p_losss, info_v_losss, info_entropys, total_losss
    return info_p_losss, info_v_losss, info_entropys

def build_occupancy_maps(state, velocity, human_states, human_velocities):  # velocity: nonholonomic_robot, human_velocities=holonomic robot
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