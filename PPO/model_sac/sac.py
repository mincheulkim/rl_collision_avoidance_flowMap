import sys

import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model_sac.net import GaussianPolicy, QNetwork_1, QNetwork_2    # default 폴더나 SAC/까지만 잡혀 있어서 model.을 추가해줌
from model_sac.net import GaussianPolicy_PED, QNetwork_1_PED, QNetwork_2_PED    # 
from model_sac.net import GaussianPolicy_MASK, QNetwork_1_MASK, QNetwork_2_MASK    # 
from model_sac.net import GaussianPolicy_CCTV, QNetwork_1_CCTV, QNetwork_2_CCTV    # 220812
from model_sac.net import GaussianPolicy_IROS, QNetwork_1_IROS, QNetwork_2_IROS    # 220819
from model_sac.net import GaussianPolicy_CCTV_HEADER, QNetwork_1_CCTV_HEADER, QNetwork_2_CCTV_HEADER    # 220822

from model_sac.utils import soft_update, hard_update
from torch.optim import Adam

import cv2 # 220816 for debugging


# 220727 아래 부분은 전혀 사용되지 않는 부분이다
hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/sac.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


### SAC
class SAC(object):   # original
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):
        ##                 3               2             2      spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     

        self.gamma = args.gamma   # 0.99
        self.tau = args.tau   # 0.005
        self.alpha = args.alpha   # 0.2

        self.policy_type = args.policy   # Gaussian
        self.target_update_interval = args.target_update_interval   # 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   # True

        self.device = torch.device("cuda" if args.cuda else "cpu")   # True

        self.action_space_array = np.array(action_space)   # paces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.action_space = action_space
        self.critic_1 = QNetwork_1(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:    # Deterministic일경우. 실행되지는 않을 듯
            self.alpha = 0
            self.automatic_entropy_tuning = False
            #self.policy = DeterministicPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy = GaussianPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state_list, evaluate=False):   # state_list = [frame_stack, goal, speed]
        frame_list, goal_list, vel_list = [], [], []
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])

            
        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)


        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()   # 220707 처음에 주석처리되 있었음        # [1, 3, 512]
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()     # main_sac 실행시 argument가 --gpu true # [1, 2]
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()                                              # [1, 2]
        
        # [1, 3, 512], [1, 2], [1, 2], [1, 3, 3, 12, 12]

        #frame_list = Variable(torch.from_numpy(frame_list)).float().cpu()  # 220707 주석처리함 cpu -> cuda로 되게
        #goal_list = Variable(torch.from_numpy(goal_list)).float().cpu()    # main_sac 실행시 argument가 --gpu false
        #vel_list = Variable(torch.from_numpy(vel_list)).float().cpu()
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list)

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list)
        
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch = memory.sample(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = mask_batch.astype(np.float32)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        '''
        print(frame_batch.shape)
        print(goal_batch.shape)
        print(speed_batch.shape)
        print(action_batch.shape)
        '''

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch)
                                                        #                [1024, 3, 512]    [1024, 2]        [1024, 2]        [1024, 3, 3, 12, 12]
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step


        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch)

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi)
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi)
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()





### SAC_PED 220720
class SAC_PED(object):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):
        ##                 3               2             2      spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     

        self.gamma = args.gamma   # 0.99
        self.tau = args.tau   # 0.005
        self.alpha = args.alpha   # 0.2

        self.policy_type = args.policy   # Gaussian
        self.target_update_interval = args.target_update_interval   # 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   # True

        self.device = torch.device("cuda" if args.cuda else "cpu")   # True

        self.action_space_array = np.array(action_space)   # paces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.action_space = action_space
        self.critic_1 = QNetwork_1_PED(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1_PED(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2_PED(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2_PED(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy_PED(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
                            # TODO. num_vel_frame_flow 추가해야 함

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:    # Deterministic일경우. 실행되지는 않을 듯
            self.alpha = 0
            self.automatic_entropy_tuning = False
            #self.policy = DeterministicPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy = GaussianPolicy_PED(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state_list, evaluate=False):   # state_list = [frame_stack, goal, speed]
        frame_list, goal_list, vel_list, ped_list = [], [], [], []
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])
            ped_list.append(i[3]) # 220712

            
        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)
        ped_list = np.asarray(ped_list)  # 220712


        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()   # 220707 처음에 주석처리되 있었음        # [1, 3, 512]
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()     # main_sac 실행시 argument가 --gpu true # [1, 2]
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()                                              # [1, 2]
        ped_list = Variable(torch.from_numpy(ped_list)).float().cuda()       # 220712                               # [1, 3, 3, 12, 12]  # 그냥1, time(t-2,t-1,t), 3(pose,v_x, vy), size
        
        # [1, 3, 512], [1, 2], [1, 2], [1, 3, 3, 12, 12]

        #frame_list = Variable(torch.from_numpy(frame_list)).float().cpu()  # 220707 주석처리함 cpu -> cuda로 되게
        #goal_list = Variable(torch.from_numpy(goal_list)).float().cpu()    # main_sac 실행시 argument가 --gpu false
        #vel_list = Variable(torch.from_numpy(vel_list)).float().cpu()
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list, ped_list)   # GaussianPolicy

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list, ped_list)   # GaussianPolicy
        
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch, ped_batch, next_ped_batch = memory.sample_ped(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = mask_batch.astype(np.float32)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        ped_batch = torch.FloatTensor(ped_batch).to(self.device)  # 220712
        next_ped_batch = torch.FloatTensor(next_ped_batch).to(self.device)  # 220712

        '''
        print(frame_batch.shape)
        print(goal_batch.shape)
        print(speed_batch.shape)
        print(action_batch.shape)
        '''

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch, next_ped_batch)
                                                        #                [1024, 3, 512]    [1024, 2]        [1024, 2]        [1024, 3, 3, 12, 12]
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, next_ped_batch)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, next_ped_batch) 

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch, ped_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch, ped_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch, ped_batch)  # 220712

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi, ped_batch)  # tod
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi, ped_batch)  # tod
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    
    
    
### SAC_MASK 220725
class SAC_MASK(object):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):
        ##                 3               2             2      spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     

        self.gamma = args.gamma   # 0.99
        self.tau = args.tau   # 0.005
        self.alpha = args.alpha   # 0.2

        self.policy_type = args.policy   # Gaussian
        self.target_update_interval = args.target_update_interval   # 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   # True

        self.device = torch.device("cuda" if args.cuda else "cpu")   # True

        self.action_space_array = np.array(action_space)   # paces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.action_space = action_space
        self.critic_1 = QNetwork_1_MASK(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1_MASK(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2_MASK(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2_MASK(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy_MASK(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
                            # TODO. num_vel_frame_flow 추가해야 함

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:    # Deterministic일경우. 실행되지는 않을 듯
            self.alpha = 0
            self.automatic_entropy_tuning = False
            #self.policy = DeterministicPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy = GaussianPolicy_MASK(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state_list, evaluate=False):   # state_list = [frame_stack, goal, speed]
        frame_list, goal_list, vel_list, mask_list = [], [], [], []
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])
            mask_list.append(i[3]) # 220712

            
        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)
        mask_list = np.asarray(mask_list)  # 220712


        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()   # 220707 처음에 주석처리되 있었음        # [1, 3, 512]
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()     # main_sac 실행시 argument가 --gpu true # [1, 2]
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()                                              # [1, 2]
        mask_list = Variable(torch.from_numpy(mask_list)).float().cuda()       # 220712                               # [1, 3, 3, 12, 12]  # 그냥1, time(t-2,t-1,t), 3(pose,v_x, vy), size
        
        # [1, 3, 512], [1, 2], [1, 2], [1, 3, 3, 12, 12]

        #frame_list = Variable(torch.from_numpy(frame_list)).float().cpu()  # 220707 주석처리함 cpu -> cuda로 되게
        #goal_list = Variable(torch.from_numpy(goal_list)).float().cpu()    # main_sac 실행시 argument가 --gpu false
        #vel_list = Variable(torch.from_numpy(vel_list)).float().cpu()
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list, mask_list)   # GaussianPolicy

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list, mask_list)   # GaussianPolicy
        
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch, masks_batch, next_masks_batch = memory.sample_mask(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = mask_batch.astype(np.float32)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        masks_batch = torch.FloatTensor(masks_batch).to(self.device)  # 220712
        next_masks_batch = torch.FloatTensor(next_masks_batch).to(self.device)  # 220712

        '''
        print(frame_batch.shape)
        print(goal_batch.shape)
        print(speed_batch.shape)
        print(action_batch.shape)
        '''

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch, next_masks_batch)
                                                        #                [1024, 3, 512]    [1024, 2]        [1024, 2]        [1024, 3, 3, 12, 12]
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, next_masks_batch)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, next_masks_batch) 

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch, masks_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch, masks_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch, masks_batch)  # 220712

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi, masks_batch)  # tod
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi, masks_batch)  # tod
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    
    
    
    
### SAC_CCTV 220812
class SAC_CCTV(object):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):
        ##                 3               2             2      spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.gamma = args.gamma   # 0.99
        self.tau = args.tau   # 0.005
        self.alpha = args.alpha   # 0.2

        self.policy_type = args.policy   # Gaussian
        self.target_update_interval = args.target_update_interval   # 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   # True

        self.device = torch.device("cuda" if args.cuda else "cpu")   # True

        self.action_space_array = np.array(action_space)   # paces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.action_space = action_space
        self.critic_1 = QNetwork_1_CCTV(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1_CCTV(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2_CCTV(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2_CCTV(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy_CCTV(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
                            # TODO. num_vel_frame_flow 추가해야 함

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:    # Deterministic일경우. 실행되지는 않을 듯
            self.alpha = 0
            self.automatic_entropy_tuning = False
            #self.policy = DeterministicPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy = GaussianPolicy_CCTV(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state_list, evaluate=False):   # state_list = [frame_stack, goal, speed]
        frame_list, goal_list, vel_list = [], [], []
        
        lidar1_list, lidar2_list, lidar3_list, lidar4_list, lidar5_list = [], [], [], [], []
        
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])
            lidar1_list.append(i[3])
            lidar2_list.append(i[4])
            lidar3_list.append(i[5])
            lidar4_list.append(i[6])
            lidar5_list.append(i[7])

            
        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)
        lidar1_list = np.asarray(lidar1_list)  # 220812
        lidar2_list = np.asarray(lidar2_list)
        lidar3_list = np.asarray(lidar3_list)
        lidar4_list = np.asarray(lidar4_list)
        lidar5_list = np.asarray(lidar5_list)
        
        lidar_visualize = False
        if lidar_visualize:
            #cctv1 = np.stack(((lidar1_list[0])*255, (lidar1_list[1])*255, (lidar1_list[2])*255), axis=0) 
            lidar = (frame_list[0]+0.5)*255
            lidar = np.uint8(lidar)
            lidar = cv2.resize(lidar, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            cv2.imshow("Robot lidar", lidar)
            cctv1 = (lidar1_list[0]+0.5)*255
            cctv1 = np.uint8(cctv1)
            cctv1 = cv2.resize(cctv1, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            cv2.imshow("CCTV1", cctv1)
            #cctv2 = (lidar2_list[0]+0.5)*255
            #cctv2 = np.uint8(cctv2)
            #cctv2 = cv2.resize(cctv2, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV2", cctv2)
            #cctv3 = (lidar3_list[0]+0.5)*255
            #cctv3 = np.uint8(cctv3)
            #cctv3 = cv2.resize(cctv3, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV3", cctv3)
            #cctv4 = (lidar4_list[0]+0.5)*255
            #cctv4 = np.uint8(cctv4)
            #cctv4 = cv2.resize(cctv4, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV4", cctv4)
            #cctv5 = (lidar5_list[0]+0.5)*255
            #cctv5 = np.uint8(cctv5)
            #cctv5 = cv2.resize(cctv5, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV5", cctv5)
            cv2.waitKey(1)   


        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()   # 220707 처음에 주석처리되 있었음        # [1, 3, 512]
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()     # main_sac 실행시 argument가 --gpu true # [1, 2]
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()                                              # [1, 2]
        lidar1_list = Variable(torch.from_numpy(lidar1_list)).float().cuda() 
        lidar2_list = Variable(torch.from_numpy(lidar2_list)).float().cuda() 
        lidar3_list = Variable(torch.from_numpy(lidar3_list)).float().cuda() 
        lidar4_list = Variable(torch.from_numpy(lidar4_list)).float().cuda() 
        lidar5_list = Variable(torch.from_numpy(lidar5_list)).float().cuda() 
        
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list, lidar1_list, lidar2_list, lidar3_list, lidar4_list, lidar5_list)   # GaussianPolicy

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list, lidar1_list, lidar2_list, lidar3_list, lidar4_list, lidar5_list)   # GaussianPolicy
        
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch, lidar1_batch, n_lidar1_batch, lidar2_batch, n_lidar2_batch, lidar3_batch, n_lidar3_batch, lidar4_batch, n_lidar4_batch, lidar5_batch, n_lidar5_batch  = memory.sample_cctv(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = mask_batch.astype(np.float32)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        lidar1_batch = torch.FloatTensor(lidar1_batch).to(self.device)
        n_lidar1_batch = torch.FloatTensor(n_lidar1_batch).to(self.device)
        lidar2_batch = torch.FloatTensor(lidar2_batch).to(self.device)
        n_lidar2_batch = torch.FloatTensor(n_lidar2_batch).to(self.device)
        lidar3_batch = torch.FloatTensor(lidar3_batch).to(self.device)
        n_lidar3_batch = torch.FloatTensor(n_lidar3_batch).to(self.device)
        lidar4_batch = torch.FloatTensor(lidar4_batch).to(self.device)
        n_lidar4_batch = torch.FloatTensor(n_lidar4_batch).to(self.device)
        lidar5_batch = torch.FloatTensor(lidar5_batch).to(self.device)
        n_lidar5_batch = torch.FloatTensor(n_lidar5_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch, n_lidar1_batch, n_lidar2_batch, n_lidar3_batch, n_lidar4_batch, n_lidar5_batch)
                                                        #                [1024, 3, 512]    [1024, 2]        [1024, 2]        [1024, 3, 3, 12, 12]
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, n_lidar1_batch, n_lidar2_batch, n_lidar3_batch, n_lidar4_batch, n_lidar5_batch)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, n_lidar1_batch, n_lidar2_batch, n_lidar3_batch, n_lidar4_batch, n_lidar5_batch)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step


        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch)

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch)
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch)
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        #       critic_1_loss,   critic_2_loss,    policy_loss,           ent_loss,          alpha
        #        loss/critic_1,  loss/critic_2,    loss/policy,           loss/entropy,      entropy_temprature/alpha
        
        
        
### SAC_CCTV_HEADER 220822 기존 CCTV에서 헤더(rel pose of cctv만 추가한거)
class SAC_CCTV_HEADER(object):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):
        ##                 3               2             2      spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.gamma = args.gamma   # 0.99
        self.tau = args.tau   # 0.005
        self.alpha = args.alpha   # 0.2

        self.policy_type = args.policy   # Gaussian
        self.target_update_interval = args.target_update_interval   # 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   # True

        self.device = torch.device("cuda" if args.cuda else "cpu")   # True

        self.action_space_array = np.array(action_space)   # paces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.action_space = action_space
        self.critic_1 = QNetwork_1_CCTV_HEADER(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1_CCTV_HEADER(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2_CCTV_HEADER(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2_CCTV_HEADER(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy_CCTV_HEADER(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:    # Deterministic일경우. 실행되지는 않을 듯
            print('error')

    def select_action(self, state_list, evaluate=False):   # state_list = [frame_stack, goal, speed]
        frame_list, goal_list, vel_list = [], [], []
        
        lidar1_list, lidar2_list, lidar3_list, lidar4_list, lidar5_list, cctv_header = [], [], [], [], [], []
        
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])
            lidar1_list.append(i[3])
            lidar2_list.append(i[4])
            lidar3_list.append(i[5])
            lidar4_list.append(i[6])
            lidar5_list.append(i[7])
            cctv_header.append(i[8])

            
        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)
        lidar1_list = np.asarray(lidar1_list)  # 220812
        lidar2_list = np.asarray(lidar2_list)
        lidar3_list = np.asarray(lidar3_list)
        lidar4_list = np.asarray(lidar4_list)
        lidar5_list = np.asarray(lidar5_list)
        cctv_header = np.asarray(cctv_header)
        
        lidar_visualize = False
        if lidar_visualize:
            #cctv1 = np.stack(((lidar1_list[0])*255, (lidar1_list[1])*255, (lidar1_list[2])*255), axis=0) 
            lidar = (frame_list[0]+0.5)*255
            lidar = np.uint8(lidar)
            lidar = cv2.resize(lidar, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            cv2.imshow("Robot lidar", lidar)
            cctv1 = (lidar1_list[0]+0.5)*255
            cctv1 = np.uint8(cctv1)
            cctv1 = cv2.resize(cctv1, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            cv2.imshow("CCTV1", cctv1)
            #cctv2 = (lidar2_list[0]+0.5)*255
            #cctv2 = np.uint8(cctv2)
            #cctv2 = cv2.resize(cctv2, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV2", cctv2)
            #cctv3 = (lidar3_list[0]+0.5)*255
            #cctv3 = np.uint8(cctv3)
            #cctv3 = cv2.resize(cctv3, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV3", cctv3)
            #cctv4 = (lidar4_list[0]+0.5)*255
            #cctv4 = np.uint8(cctv4)
            #cctv4 = cv2.resize(cctv4, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV4", cctv4)
            #cctv5 = (lidar5_list[0]+0.5)*255
            #cctv5 = np.uint8(cctv5)
            #cctv5 = cv2.resize(cctv5, dsize=(512,256), interpolation=cv2.INTER_NEAREST)  
            #cv2.imshow("CCTV5", cctv5)
            cv2.waitKey(1)   


        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()   # 220707 처음에 주석처리되 있었음        # [1, 3, 512]
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()     # main_sac 실행시 argument가 --gpu true # [1, 2]
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()                                              # [1, 2]
        lidar1_list = Variable(torch.from_numpy(lidar1_list)).float().cuda() 
        lidar2_list = Variable(torch.from_numpy(lidar2_list)).float().cuda() 
        lidar3_list = Variable(torch.from_numpy(lidar3_list)).float().cuda() 
        lidar4_list = Variable(torch.from_numpy(lidar4_list)).float().cuda() 
        lidar5_list = Variable(torch.from_numpy(lidar5_list)).float().cuda() 
        cctv_header = Variable(torch.from_numpy(cctv_header)).float().cuda()
        
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list, lidar1_list, lidar2_list, lidar3_list, lidar4_list, lidar5_list, cctv_header)   # GaussianPolicy

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list, lidar1_list, lidar2_list, lidar3_list, lidar4_list, lidar5_list, cctv_header)   # GaussianPolicy
        
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory  220822 cctv_header_batch 추가
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch, lidar1_batch, n_lidar1_batch, lidar2_batch, n_lidar2_batch, lidar3_batch, n_lidar3_batch, lidar4_batch, n_lidar4_batch, lidar5_batch, n_lidar5_batch, cctv_header_batch  = memory.sample_cctv_header(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = mask_batch.astype(np.float32)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        lidar1_batch = torch.FloatTensor(lidar1_batch).to(self.device)
        n_lidar1_batch = torch.FloatTensor(n_lidar1_batch).to(self.device)
        lidar2_batch = torch.FloatTensor(lidar2_batch).to(self.device)
        n_lidar2_batch = torch.FloatTensor(n_lidar2_batch).to(self.device)
        lidar3_batch = torch.FloatTensor(lidar3_batch).to(self.device)
        n_lidar3_batch = torch.FloatTensor(n_lidar3_batch).to(self.device)
        lidar4_batch = torch.FloatTensor(lidar4_batch).to(self.device)
        n_lidar4_batch = torch.FloatTensor(n_lidar4_batch).to(self.device)
        lidar5_batch = torch.FloatTensor(lidar5_batch).to(self.device)
        n_lidar5_batch = torch.FloatTensor(n_lidar5_batch).to(self.device)
        cctv_header_batch = torch.FloatTensor(cctv_header_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch, n_lidar1_batch, n_lidar2_batch, n_lidar3_batch, n_lidar4_batch, n_lidar5_batch, cctv_header_batch)
                                                        #                [1024, 3, 512]    [1024, 2]        [1024, 2]        [1024, 3, 3, 12, 12]
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, n_lidar1_batch, n_lidar2_batch, n_lidar3_batch, n_lidar4_batch, n_lidar5_batch, cctv_header_batch)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, n_lidar1_batch, n_lidar2_batch, n_lidar3_batch, n_lidar4_batch, n_lidar5_batch, cctv_header_batch)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch, cctv_header_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch, cctv_header_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step


        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch, cctv_header_batch)

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch, cctv_header_batch)
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi, lidar1_batch, lidar2_batch, lidar3_batch, lidar4_batch, lidar5_batch, cctv_header_batch)
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        
        
### SAC_IROS 220819 from SAC_ped
class SAC_IROS(object):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, action_space, args):
        ##                 3               2             2      spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     

        self.gamma = args.gamma   # 0.99
        self.tau = args.tau   # 0.005
        self.alpha = args.alpha   # 0.2

        self.policy_type = args.policy   # Gaussian
        self.target_update_interval = args.target_update_interval   # 1
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   # True

        self.device = torch.device("cuda" if args.cuda else "cpu")   # True

        self.action_space_array = np.array(action_space)   # paces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0])     
        self.action_space = action_space
        self.critic_1 = QNetwork_1_IROS(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)

        self.critic_1_target = QNetwork_1_IROS(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_1_target, self.critic_1)

        self.critic_2 = QNetwork_2_IROS(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)

        self.critic_2_target = QNetwork_2_IROS(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_2_target, self.critic_2)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space_array.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy_IROS(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:    # Deterministic일경우. 실행되지는 않을 듯
            self.alpha = 0
            self.automatic_entropy_tuning = False
            #self.policy = DeterministicPolicy(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy = GaussianPolicy_IROS(num_frame_obs, num_goal_obs, num_vel_obs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            #원래는 위에 determinsitic이 주석풀고ㄷ 되야 함
            
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state_list, evaluate=False):   # state_list = [frame_stack, goal, speed]
        frame_list, goal_list, vel_list, ped_list = [], [], [], []
        for i in state_list:
            frame_list.append(i[0])
            goal_list.append(i[1])
            vel_list.append(i[2])
            ped_list.append(i[3]) # 220712

            
        frame_list = np.asarray(frame_list)
        goal_list = np.asarray(goal_list)
        vel_list = np.asarray(vel_list)
        ped_list = np.asarray(ped_list)  # 220712


        frame_list = Variable(torch.from_numpy(frame_list)).float().cuda()   # 220707 처음에 주석처리되 있었음        # [1, 3, 512]
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()     # main_sac 실행시 argument가 --gpu true # [1, 2]
        vel_list = Variable(torch.from_numpy(vel_list)).float().cuda()                                              # [1, 2]
        ped_list = Variable(torch.from_numpy(ped_list)).float().cuda()       # 220712                               # [1, 3, 3, 12, 12]  # 그냥1, time(t-2,t-1,t), 3(pose,v_x, vy), size
        
        # [1, 3, 512], [1, 2], [1, 2], [1, 3, 3, 12, 12]

        #frame_list = Variable(torch.from_numpy(frame_list)).float().cpu()  # 220707 주석처리함 cpu -> cuda로 되게
        #goal_list = Variable(torch.from_numpy(goal_list)).float().cpu()    # main_sac 실행시 argument가 --gpu false
        #vel_list = Variable(torch.from_numpy(vel_list)).float().cpu()
        #state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(frame_list, goal_list, vel_list, ped_list)   # GaussianPolicy

        else:
            _, _, action = self.policy.sample(frame_list, goal_list, vel_list, ped_list)   # GaussianPolicy
        
        #return action.detach().cpu().numpy()[0]
        return action.data.cpu().numpy()


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        frame_batch, goal_batch, speed_batch, action_batch, reward_batch, next_frame_batch, next_goal_batch, next_speed_batch, mask_batch, ped_batch, next_ped_batch = memory.sample_iros(batch_size=batch_size)

        frame_batch = torch.FloatTensor(frame_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        speed_batch = torch.FloatTensor(speed_batch).to(self.device)
        next_frame_batch = torch.FloatTensor(next_frame_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        next_speed_batch = torch.FloatTensor(next_speed_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = mask_batch.astype(np.float32)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        ped_batch = torch.FloatTensor(ped_batch).to(self.device)  # 220712
        next_ped_batch = torch.FloatTensor(next_ped_batch).to(self.device)  # 220712

        '''
        print(frame_batch.shape)
        print(goal_batch.shape)
        print(speed_batch.shape)
        print(action_batch.shape)
        '''

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_frame_batch, next_goal_batch, next_speed_batch, next_ped_batch)
                                                        #                [1024, 3, 512]    [1024, 2]        [1024, 2]        [1024, 3, 3, 12, 12]
            qf1_next_target = self.critic_1_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, next_ped_batch)
            qf2_next_target = self.critic_2_target(next_frame_batch, next_goal_batch, next_speed_batch, next_state_action, next_ped_batch) 

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * (min_qf_next_target)

        qf1 = self.critic_1(frame_batch, goal_batch, speed_batch, action_batch, ped_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf2 = self.critic_2(frame_batch, goal_batch, speed_batch, action_batch, ped_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value)  

        self.critic_1_optim.zero_grad()
        qf1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        qf2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(frame_batch, goal_batch, speed_batch, ped_batch)  # 220712

        qf1_pi = self.critic_1(frame_batch, goal_batch, speed_batch, pi, ped_batch)  # tod
        qf2_pi = self.critic_2(frame_batch, goal_batch, speed_batch, pi, ped_batch)  # tod
    
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    
    