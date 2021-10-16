import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from model.utils import log_normal_density

import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3



class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.logstd = nn.Parameter(torch.zeros(2))
        #self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)    # feature -> 2 (actor)
        self.fc_v  = nn.Linear(256,1)    # feature -> 1 (critic)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.act_fea_cv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1)   # frame = 3
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)     # 4096->256
        self.act_fc2 =  nn.Linear(256+2+2, 256)   # 260-> 256

        self.actor1 = nn.Linear(256, 1)
        self.actor2 = nn.Linear(256, 1)

    def pi(self, x, softmax_dim = 0):
        s_list, goal_list, speed_list = [], [], []

        s_list = x[0]
        goal_list = x[1]
        speed_list = x[2]
             
        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = torch.from_numpy(s_list).unsqueeze(dim=0).float()
        goal_list = torch.from_numpy(goal_list).unsqueeze(dim=0).float()
        speed_list = torch.from_numpy(speed_list).unsqueeze(dim=0).float()

        #x = F.relu(self.fc1(x))
        a = F.relu(self.act_fea_cv1(s_list))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal_list, speed_list), dim=-1)
        a = F.relu(self.act_fc2(a))    # 260 -> 256

        x = self.fc_pi(a)    # 256 -> 2

        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        #print('action:',action)

        prob = F.softmax(x, dim=softmax_dim)
        return prob, action

    def pi_batch(self, s1, s2, s3, softmax_dim = 0):
        s_list, goal_list, speed_list = [], [], []

        s_list = s1
        goal_list = s2
        speed_list = s3
             
        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = torch.from_numpy(s_list).float()
        goal_list = torch.from_numpy(goal_list).float()
        speed_list = torch.from_numpy(speed_list).float()

        s_list=torch.squeeze(s_list, dim=0)
        goal_list=torch.squeeze(goal_list, dim=0)
        speed_list=torch.squeeze(speed_list, dim=0)

        #x = F.relu(self.fc1(x))
        a = F.relu(self.act_fea_cv1(s_list))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal_list, speed_list), dim=-1)
        a = F.relu(self.act_fc2(a))    # 260 -> 256

        x = self.fc_pi(a)    # 256 -> 2
        prob = F.softmax(x, dim=softmax_dim)

        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        return prob, action
    
    def v(self, x, y, z):
        #x = F.relu(self.fc1(x))
        s_list, goal_list, speed_list = [], [], []

        s_list = x
        goal_list = y
        speed_list = z
             
        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = torch.from_numpy(s_list).float()
        goal_list = torch.from_numpy(goal_list).float()
        speed_list = torch.from_numpy(speed_list).float()


        #x = F.relu(self.fc1(x))
        s_list=torch.squeeze(s_list, dim=0)
        goal_list=torch.squeeze(goal_list, dim=0)
        speed_list=torch.squeeze(speed_list, dim=0)

        a = F.relu(self.act_fea_cv1(s_list))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal_list, speed_list), dim=-1)
        a = F.relu(self.act_fc2(a))    # 260 -> 256

        v = self.fc_v(a)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):   # 0,1,2,3
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
        v_batch, occupancy_maps_batch = [], [], [], [], [], [], [], [], []
        s_temp, goal_temp, speed_temp = [], [], []
        s_prime_temp, goal_prime_temp, speed_prime_temp = [], [], []
        s_prime_batch, goal_prime_batch, speed_prime_batch=[],[],[]

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_temp.append(s[0])   # 1. lidar
            goal_temp.append(s[1])   # 2. local goal
            speed_temp.append(s[2])   # 3. velocity

            s_batch.append(s_temp)
            goal_batch.append(goal_temp)
            speed_batch.append(speed_temp)

            s_temp = []
            goal_temp = []
            speed_temp = []

            a_batch.append(a)   # A
            r_batch.append(r)   # reward

            s_prime_temp.append(s_prime[0])   # 1. lidar
            goal_prime_temp.append(s_prime[1])   # 2. local goal
            speed_prime_temp.append(s_prime[2])   # 3. velocity

            s_prime_batch.append(s_prime_temp)
            goal_prime_batch.append(goal_prime_temp)
            speed_prime_batch.append(speed_prime_temp)

            s_prime_temp = []
            goal_prime_temp = []
            speed_prime_temp = []

            l_batch.append(prob_a)   # logprob
            d_batch.append(done)   # V
                  
        s1=torch.tensor(s_batch, dtype=torch.float)
        s2=torch.tensor(goal_batch, dtype=torch.float)
        s3=torch.tensor(speed_batch, dtype=torch.float)
        s11=torch.tensor(s_prime_batch, dtype=torch.float)
        s22=torch.tensor(goal_prime_batch, dtype=torch.float)
        s33=torch.tensor(speed_prime_batch, dtype=torch.float)
        a=torch.tensor(a_batch)
        r= torch.tensor(r_batch)
        s_prime= torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask=torch.tensor(d_batch, dtype=torch.float)
        prob_a=torch.tensor(l_batch)


        self.data = []
        #return s, a, r, s_prime, done_mask, prob_a
        return s1, s2, s3, a, r, s11, s22, s33, done_mask, prob_a
        
    def train_net(self):
        #s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        s1, s2, s3, a, r, s11, s22, s33, done_mask, prob_a = self.make_batch()
        #s,               s_prime
        a=torch.unsqueeze(a, dim=0)

        for i in range(K_epoch):
            #td_target = r + gamma * self.v(s_prime) * done_mask
            td_target = r + gamma * self.v(s11, s22, s33) * done_mask
            delta = td_target - self.v(s1, s2, s3)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi_batch(s1, s2, s3, softmax_dim=1)
            pi_a = pi.gather(1,a)

            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            #loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s1, s2, s3) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):    # frames= 3, action_space= 2    from ppo_stage3.py/main()
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)


        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)



    def forward(self, x, goal, speed):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)


        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        v, _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):   # obs_space= 512, action_space= 2, called from ppo_stage3.py
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        self.mu.weight.data.mul_(0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_fc3.weight.data.mul(0.1)

    def forward(self, x):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        return v, action, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

###########  RVO Policy ################
########################################

class RVOPolicy(nn.Module):
    def __init__(self, frames, action_space):    # frames= 3, action_space= 2    from ppo_stage3.py/main()
        super(RVOPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)


        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)

        ### incorporate ORCA 211020
        
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def forward(self, x, goal, speed, pose):   # now create velocity
        """
            parameter
                x: 
                (local)goal: tensor([[0,0],[0,0],[0,0],[0,0],[0,0]], device='cuda:0')
                speed: tensor([[0,0],[0,0],[0,0],[0,0],[0,0]], device='cuda:0')
            returns value estimation, action, log_action_prob

        """
        # action
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        #v, _, _, mean = self.forward(x, goal, speed)
        v, _, _, mean = self.forward(x, goal, speed, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

###########  Robot Policy ################
########################################

class RobotPolicy(nn.Module):
    def __init__(self, frames, action_space):    # frames= 3, action_space= 2    from ppo_stage3.py/main()
        super(RobotPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))
        # in_channel: input's feature dimension, out_channel: want output dimension, kernel_size = frame size, stride = how moving
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)   # frame = 3
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)     # 4096->256
        self.act_fc2 =  nn.Linear(256+2+2, 128)   # 260-> 128
        self.actor1 = nn.Linear(128, 1)             
        self.actor2 = nn.Linear(128, 1)


        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)   # frame = 3
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, goal, speed, p_list):   # now create velocity
        """
            parameter
                x: 
                (local)goal: tensor([[0,0],[0,0],[0,0],[0,0],[0,0]], device='cuda:0')
                speed: tensor([[0,0],[0,0],[0,0],[0,0],[0,0]], device='cuda:0')
            returns value estimation, action, log_action_prob
        """
        
        #print('lidar:',x[:,1,0],x[:,1,100],x[:,1,200],x[:,1,300],x[:,1,400],x[:,1,500])
        #print('goal:',goal)
        #print('speed:',speed)
        #print('pose_robot:',p_list)

        # action
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))          # constrained the mean of traslational velocity in (0.0, 1.0)   sigmoid  linear vel
        mean2 = F.tanh(self.actor2(a))             # constrained the mean of rotational velocity in (-1.0, 1.0)    tanh     angular vel
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)    # sampled from Gaussian dstribution (v_mean, v_logstd)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        #v, _, _, mean = self.forward(x, goal, speed)
        v, _, _, mean = self.forward(x, goal, speed, speed)    # batch 128 input
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy        

##############  211101   ###############
#####  RVO Policy with Local Map #######
########################################

class RobotPolicy_LM(nn.Module):
    def __init__(self, frames, action_space):    # frames= 3, action_space= 2    from ppo_stage3.py/main()
        super(RobotPolicy_LM, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        # nn.Conv1D description
        # in_channels: input's feature dimension, 3
        # out_channels: how i want to output dimension
        # kernel_size : how much see time step(=frame_size=filter_size)
        # stride: how much moving kernel to see
        # 1. for actor net
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)   # lidar
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)       # lidar
        self.act_fc1 = nn.Linear(128*32, 256)     # lidar.fc
        self.act_LM = nn.Linear(48, 16)    # localmap
        #self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.act_fc2 =  nn.Linear(256+2+2+16, 128)  # w localmap, 211104
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        # 2. for critic net
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_LM = nn.Linear(48, 16)    # localmap 
        #self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.crt_fc2 = nn.Linear(256+2+2+16, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, goal, speed, occupancy_map):   # with localmap
        """
            parameter
                x: s_list(lidar)  (1,3,512)
                (local)goal: tensor([[0,0],[0,0],[0,0],[0,0],[0,0]], device='cuda:0')   (1,2)
                speed: tensor([[0,0],[0,0],[0,0],[0,0],[0,0]], device='cuda:0')         (1,2)
                localmap: (1,48)

            returns value estimation, action, log_action_prob
        """
        '''
        print('x:',x)
        print('goal:',goal)
        print('speed:',speed)
        print('LM:',occupancy_map)
        '''

        # 1. action(policy)\
        a = F.relu(self.act_fea_cv1(x))   # (1,3,512) -> (1,32,255)
        a = F.relu(self.act_fea_cv2(a))   # (1,3,255) -> (1,32,128)
        a = a.view(a.shape[0], -1)        # (1,32,128) -> (1,4096)
        a = F.relu(self.act_fc1(a))       # (1,4096) -> (1, 256)

        # TODO. concat LocalMap 
        lm_a = F.relu(self.act_LM(occupancy_map))   # (1,48) -> (1,16)
        #TODO a = torch.cat((a, goal, speed, localmap), dim=-1)   # concat localmap
        #a = torch.cat((a, goal, speed), dim=-1)   # concat feature lidar, local goal, speed
        
        a = torch.cat((a, goal, speed, lm_a), dim=-1)   # concat feature lidar, local goal, speed and local map # 211104
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))   # 0~1    # linear vel
        mean2 = F.tanh(self.actor2(a))      # -1~1   # angular vel
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # 2. value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))

        # TODO. concat LocalMap 
        #TODO v = torch.cat((v, goal, speed, localmap), dim=-1)   # concat localmap
        lm_c = F.relu(self.crt_LM(occupancy_map))   # (1,48) -> (1,16)
        #v = torch.cat((v, goal, speed), dim=-1)
        v = torch.cat((v, goal, speed, lm_c), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)


        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action, occupancy_map):

        #v, _, _, mean = self.forward(x, goal, speed)
        #v, _, _, mean = self.forward(x, goal, speed, speed)
        v, _, _, mean = self.forward(x, goal, speed, occupancy_map)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy        


if __name__ == '__main__':
    from torch.autograd import Variable

    net = MLPPolicy(3, 2)

    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print(v)

