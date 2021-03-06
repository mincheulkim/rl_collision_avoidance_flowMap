import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from model.utils import log_normal_density

import rvo2  # 211020

class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)



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

    def evaluate_actions(self, x, goal, speed, action):    # for robot
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

