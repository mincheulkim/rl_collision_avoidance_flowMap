import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import Normal

from model.utils import log_normal_density


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 1,  -1)

class QNetwork_1(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_1, self).__init__()
        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # Q1 architecture
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, frame, goal, vel, action):
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        '''
        print(frame.shape)
        print(goal.shape)
        print(vel.shape)
        print(action.shape)
        '''
        xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        return x1

class QNetwork_2(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_2, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # Q2 architecture
        #self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + num_actions, hidden_dim)
        self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, frame, goal, vel, action):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))

        xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x2

class ValueNetwork(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, frame, goal, vel):
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))

        state = torch.cat([o1, goal, vel], 1) # observation

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class GaussianPolicy(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim, action_space=None):
                       #    3              2            2          
        super(GaussianPolicy, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(2))

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions) # Different from PPO

        self.mean1_linear = nn.Linear(hidden_dim, 1) # Different from PPO
        self.mean2_linear = nn.Linear(hidden_dim, 1) # Different from PPO


        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            
            scale = [0.5, 1]
            bias = [0.5, 0]

            self.action_scale = torch.FloatTensor(scale)
            self.action_bias = torch.FloatTensor(bias)

            print("self.action_scale: ", self.action_scale)
            print("self.action_bias: ", self.action_bias)

    def forward(self, frame, goal, vel):
        o1 = F.relu(self.fea_cv1(frame))   
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)   # (1, 4096)
        o1 = F.relu(self.fc1(o1))   # (1, 256)
        # concat lidar_frame, local_goal, velocity
        state = torch.cat((o1, goal, vel), dim=-1) # observation   # (1, 260)


        x = F.relu(self.linear1(state))   # (1, 256)
        x = F.relu(self.linear2(x))   # (1, 256)
        #mean1 = F.sigmoid(self.mean1_linear(x))
        #mean2 = F.tanh(self.mean2_linear(x))
        
        #mean = torch.cat((mean1, mean2), dim=-1)

        mean = self.mean_linear(x)   # (1, 2)

        #log_std = self.logstd.expand_as(mean)

        log_std = self.log_std_linear(x)   # (1, 2)
        
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, frame, goal, vel):   # GaussianPolicy.sample(): 
        '''
        Input: frame_list, goal_list, vel_list
        Output: action(training), log_prob, mean(evaluate)
        '''
        mean, log_std = self.forward(frame, goal, vel) 
        std = log_std.exp()   # e^log_std
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)


        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale  + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


if __name__ == '__main__':
    from torch.autograd import Variable





#################  PED MAP  ###############################

class QNetwork_1_PED(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_1_PED, self).__init__()
        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # 220713 ADDED for Conv2D for ped_map
        self.ped_cv1 = nn.Conv3d(in_channels=2, out_channels=16, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_cv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size = (1, 2, 2), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_fc1 = nn.Linear(32*13*13, 256)

        # Q1 architecture
        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256, hidden_dim)   # 220713 ped_map
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel, action):
    def forward(self, frame, goal, vel, action, ped):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        '''
        print(frame.shape)
        print(goal.shape)
        print(vel.shape)
        print(action.shape)
        '''
        #220713 ADDED Ped_map feature extract
        ped = ped.permute(0, 2, 1, 3, 4)
        p1 = F.relu(self.ped_cv1(ped))
        p1 = F.relu(self.ped_cv2(p1))
        p1 = p1.view(p1.shape[0], -1) 
        p1 = F.relu(self.ped_fc1(p1))

        #xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        xu = torch.cat((o1, goal, vel, action, p1), dim=-1) # 220713 observation + action + ped_map
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

class QNetwork_2_PED(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_2_PED, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # 220713 ADDED for Conv2D for ped_map
        self.ped_cv1 = nn.Conv3d(in_channels=2, out_channels=16, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_cv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size = (1, 2, 2), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_fc1 = nn.Linear(32*13*13, 256)

        # Q2 architecture
        #self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + num_actions, hidden_dim)
        #self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256, hidden_dim) # 220713 ped_map
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel, action):
    def forward(self, frame, goal, vel, action, ped):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        
        #220713 ADDED Ped_map feature extract
        ped = ped.permute(0, 2, 1, 3, 4)
        p1 = F.relu(self.ped_cv1(ped))
        p1 = F.relu(self.ped_cv2(p1))
        p1 = p1.view(p1.shape[0], -1) 
        p1 = F.relu(self.ped_fc1(p1))

        #xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        xu = torch.cat((o1, goal, vel, action, p1), dim=-1) # 220713 observation + action + ped_map
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x2

class ValueNetwork_PED(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, hidden_dim):
        super(ValueNetwork_PED, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # 220713 ADDED for Conv2D for ped_map
        self.ped_cv1 = nn.Conv3d(in_channels=2, out_channels=16, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_cv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size = (1, 2, 2), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_fc1 = nn.Linear(32*13*13, 256)


        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + 256, hidden_dim)  # 220713 ped_map
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel):
    def forward(self, frame, goal, vel, ped):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        
        #220713 ADDED Ped_map feature extract
        ped = ped.permute(0, 2, 1, 3, 4)
        p1 = F.relu(self.ped_cv1(ped))
        p1 = F.relu(self.ped_cv2(p1))
        p1 = p1.view(p1.shape[0], -1) 
        p1 = F.relu(self.ped_fc1(p1))

        #state = torch.cat([o1, goal, vel], 1) # observation
        state = torch.cat([o1, goal, vel, p1], 1) # observation 220713 ped_map


        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class GaussianPolicy_PED(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim, action_space=None):
                       #    3              2            2          
        super(GaussianPolicy_PED, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(2))

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + 256, hidden_dim)  # 220713 ped_map
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions) # Different from PPO

        self.mean1_linear = nn.Linear(hidden_dim, 1) # Different from PPO
        self.mean2_linear = nn.Linear(hidden_dim, 1) # Different from PPO

        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        ## 220713 ADDED for Conv3D for ped_map  https://stackoverflow.com/questions/57484508/how-does-one-use-3d-convolutions-on-standard-3-channel-images
        self.ped_cv1 = nn.Conv3d(in_channels=2, out_channels=16, kernel_size = (3, 3, 3), stride = (1, 1, 1), padding = (0, 1, 1))
        self.ped_cv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size = (1, 2, 2), stride = (1, 1, 1), padding = (0, 1, 1))
        # kerner_size 3으로 시작 의미: t-2, t-1, t 세개를 하나로 묶음. 따라서 뒤의 cv2에서는 1로 시작
        self.ped_fc1 = nn.Linear(32*13*13, 256)
        '''
        N   For mini batch (or how many sequences do we want to feed at one go)       배치: 1 ro 1024
        Cin For the number of channels in our input (if our image is rgb, this is 3)  채널: 2
        D   For depth or in other words the number of images/frames in one input sequence (if we are dealing videos, this is the number of frames)  시퀀스: 3
        H   For the height of the image/frame   # 60
        W   For the width of the image/frame    # 60
        따라서 1 or 1024 x 2 x 3 x 60 x 60이 되야 함
        1) Cin: 채널 수 as 2 (vx, vy)
        2) D: 뎁스 as 3 (t-2, t-1, t)
        '''
    
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            
            scale = [0.5, 1]
            bias = [0.5, 0]

            self.action_scale = torch.FloatTensor(scale)
            self.action_bias = torch.FloatTensor(bias)

            print("self.action_scale: ", self.action_scale)
            print("self.action_bias: ", self.action_bias)

    #def forward(self, frame, goal, vel):
    def forward(self, frame, goal, vel, ped):  # 220712
        o1 = F.relu(self.fea_cv1(frame))   
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)   # (1, 4096)
        o1 = F.relu(self.fc1(o1))   # (1, 256)
        
        #220713 ADDED Ped_map feature extract
        #print('ped shape:',ped.shape)   # ([1, 3, 2, 12, 12])  # [1024, 3, 2, 12, 12]
        ped = ped.permute(0, 2, 1, 3, 4)   # N, Cin, D, H, W as 1or1024, 2, 3, 12, 12   # 차원순서바꾸기 https://startnow95.tistory.com/17
        p1 = F.relu(self.ped_cv1(ped))
        #print('after ped feature cv1:',p1.shape)   # ([1, 16, 1, 12, 12])  or [1024, 16, 1, 12, 12](batch parameter update)
        p1 = F.relu(self.ped_cv2(p1))

        #print('after ped feature cv2:',p1.shape)   # ([1, 32, 1, 13, 13])
        p1 = p1.view(p1.shape[0], -1)      # [1, 5408] or [1024, 5408]
        p1 = F.relu(self.ped_fc1(p1))
        
        # concat lidar_frame, local_goal, velocity
        #state = torch.cat((o1, goal, vel), dim=-1) # observation   # (1, 260)
        state = torch.cat((o1, goal, vel, p1), dim=-1) # observation   # 220713 ped_map

        x = F.relu(self.linear1(state))   # (1, 256)
        x = F.relu(self.linear2(x))   # (1, 256)
        #mean1 = F.sigmoid(self.mean1_linear(x))
        #mean2 = F.tanh(self.mean2_linear(x))
        
        #mean = torch.cat((mean1, mean2), dim=-1)

        mean = self.mean_linear(x)   # (1, 2)

        #log_std = self.logstd.expand_as(mean)

        log_std = self.log_std_linear(x)   # (1, 2)
        
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, frame, goal, vel, ped):   # 220712
        '''
        Input: frame_list, goal_list, vel_list
        Output: action(training), log_prob, mean(evaluate)
        '''
        mean, log_std = self.forward(frame, goal, vel, ped)   # 220712 여기에 로컬 맵 input
        std = log_std.exp()   # e^log_std
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)


        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale  + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_PED, self).to(device)


if __name__ == '__main__':
    from torch.autograd import Variable




#################  MASK 220725 ###############################

class QNetwork_1_MASK(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_1_MASK, self).__init__()
        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # TODO
        self.fea_cv1_mask = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2_mask = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1_mask = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        
        # Q1 architecture
        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256, hidden_dim)   
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel, action):
    def forward(self, frame, goal, vel, action, mask):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        '''
        print(frame.shape)
        print(goal.shape)
        print(vel.shape)
        print(action.shape)
        '''
        # TODO
        p1 = F.relu(self.fea_cv1_mask(mask))
        p1 = F.relu(self.fea_cv2_mask(p1))
        p1 = p1.view(p1.shape[0], -1)
        p1 = F.relu(self.fc1_mask(p1))
        
        #xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        xu = torch.cat((o1, goal, vel, action, p1), dim=-1) # 220713 observation + action + ped_map
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

class QNetwork_2_MASK(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_2_MASK, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # TODO
        self.fea_cv1_mask = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2_mask = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1_mask = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity


        # Q2 architecture
        #self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + num_actions, hidden_dim)
        #self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions, hidden_dim)
        self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256, hidden_dim) # 220713 ped_map
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel, action):
    def forward(self, frame, goal, vel, action, mask):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        
        # TODO
        p1 = F.relu(self.fea_cv1_mask(mask))
        p1 = F.relu(self.fea_cv2_mask(p1))
        p1 = p1.view(p1.shape[0], -1)
        p1 = F.relu(self.fc1_mask(p1))

        #xu = torch.cat((o1, goal, vel, action), dim=-1) # observation + action
        xu = torch.cat((o1, goal, vel, action, p1), dim=-1) # 220713 observation + action + ped_map
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x2

class ValueNetwork_MASK(nn.Module):   # SAC 두번째 논문부터는 value network없이 Q-network로만 구현
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, hidden_dim):
        super(ValueNetwork_MASK, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity

        # TODO
        self.fea_cv1_mask = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2_mask = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1_mask = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity


        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + 256, hidden_dim)  # 220713 ped_map
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel):
    def forward(self, frame, goal, vel, mask):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        
        # TODO
        p1 = F.relu(self.fea_cv1_mask(mask))
        p1 = F.relu(self.fea_cv2_mask(p1))
        p1 = p1.view(p1.shape[0], -1)
        p1 = F.relu(self.fc1_mask(p1))
        

        #state = torch.cat([o1, goal, vel], 1) # observation
        state = torch.cat([o1, goal, vel, p1], 1) # observation 220713 ped_map


        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class GaussianPolicy_MASK(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim, action_space=None):
                       #    3              2            2          
        super(GaussianPolicy_MASK, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(2))

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        self.fea_cv1_mask = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2_mask = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1_mask = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        
        
        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + 256, hidden_dim) 
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions) # Different from PPO

        self.mean1_linear = nn.Linear(hidden_dim, 1) # Different from PPO
        self.mean2_linear = nn.Linear(hidden_dim, 1) # Different from PPO

        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            
            scale = [0.5, 1]
            bias = [0.5, 0]

            self.action_scale = torch.FloatTensor(scale)
            self.action_bias = torch.FloatTensor(bias)

            print("self.action_scale: ", self.action_scale)
            print("self.action_bias: ", self.action_bias)

    #def forward(self, frame, goal, vel):
    def forward(self, frame, goal, vel, mask):  # 220712
        o1 = F.relu(self.fea_cv1(frame))   
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)   # (1, 4096)
        o1 = F.relu(self.fc1(o1))   # (1, 256)
        
        
        # TODO
        p1 = F.relu(self.fea_cv1_mask(mask))   
        p1 = F.relu(self.fea_cv2_mask(p1))
        p1 = p1.view(p1.shape[0], -1)   # (1, 4096)
        p1 = F.relu(self.fc1_mask(p1))   # (1, 256)
        
        # concat lidar_frame, local_goal, velocity
        #state = torch.cat((o1, goal, vel), dim=-1) # observation   # (1, 260)
        state = torch.cat((o1, goal, vel, p1), dim=-1) # observation  

        x = F.relu(self.linear1(state))   # (1, 256)
        x = F.relu(self.linear2(x))   # (1, 256)
        #mean1 = F.sigmoid(self.mean1_linear(x))
        #mean2 = F.tanh(self.mean2_linear(x))
        
        #mean = torch.cat((mean1, mean2), dim=-1)

        mean = self.mean_linear(x)   # (1, 2)

        #log_std = self.logstd.expand_as(mean)

        log_std = self.log_std_linear(x)   # (1, 2)
        
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, frame, goal, vel, mask):   # 220712
        '''
        Input: frame_list, goal_list, vel_list
        Output: action(training), log_prob, mean(evaluate)
        '''
        mean, log_std = self.forward(frame, goal, vel, mask)   # 220712 여기에 로컬 맵 input
        std = log_std.exp()   # e^log_std
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)


        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale  + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_MASK, self).to(device)


if __name__ == '__main__':
    from torch.autograd import Variable




#################  CCTV 220812 ###############################

class QNetwork_1_CCTV(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_1_CCTV, self).__init__()
        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        # 220812 lidar
        self.lidar1_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar1_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar1_fc1 = nn.Linear(128*32, 256)
        '''
        self.lidar2_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar2_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar2_fc1 = nn.Linear(128*32, 256)
        self.lidar3_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar3_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar3_fc1 = nn.Linear(128*32, 256)
        self.lidar4_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar4_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar4_fc1 = nn.Linear(128*32, 256)
        self.lidar5_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar5_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar5_fc1 = nn.Linear(128*32, 256)
        '''
       
        # Q1 architecture
        '''
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256*5, hidden_dim)  # 5 CCTVs
        '''
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256, hidden_dim)  # 1 CCTV
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel, action):
    def forward(self, frame, goal, vel, action, lidar1, lidar2, lidar3, lidar4, lidar5):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        
        # 220812 lidar preprocessing
        l1 = F.relu(self.lidar1_cv1(lidar1))   
        l1 = F.relu(self.lidar1_cv2(l1))
        l1 = l1.view(l1.shape[0], -1)   # (1, 4096)
        l1 = F.relu(self.lidar1_fc1(l1))   # (1, 256)
        '''
        l2 = F.relu(self.lidar2_cv1(lidar2))   
        l2 = F.relu(self.lidar2_cv2(l2))
        l2 = l2.view(l2.shape[0], -1)   # (1, 4096)
        l2 = F.relu(self.lidar2_fc1(l2))   # (1, 256)
        l3 = F.relu(self.lidar3_cv1(lidar3))   
        l3 = F.relu(self.lidar3_cv2(l3))
        l3 = l3.view(l3.shape[0], -1)   # (1, 4096)
        l3 = F.relu(self.lidar3_fc1(l3))   # (1, 256)
        l4 = F.relu(self.lidar4_cv1(lidar4))   
        l4 = F.relu(self.lidar4_cv2(l4))
        l4 = l4.view(l4.shape[0], -1)   # (1, 4096)
        l4 = F.relu(self.lidar4_fc1(l4))   # (1, 256)
        l5 = F.relu(self.lidar5_cv1(lidar5))   
        l5 = F.relu(self.lidar5_cv2(l5))
        l5 = l5.view(l5.shape[0], -1)   # (1, 4096)
        l5 = F.relu(self.lidar5_fc1(l5))   # (1, 256)
        '''
        '''
        xu = torch.cat((o1, goal, vel, action, l1, l2, l3, l4, l5), dim=-1) # 5 CCTVs
        '''
        xu = torch.cat((o1, goal, vel, action, l1), dim=-1) # 1 CCTV
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1

class QNetwork_2_CCTV(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim):
        super(QNetwork_2_CCTV, self).__init__()

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        # 220812 lidar
        self.lidar1_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar1_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar1_fc1 = nn.Linear(128*32, 256)
        '''
        self.lidar2_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar2_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar2_fc1 = nn.Linear(128*32, 256)
        self.lidar3_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar3_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar3_fc1 = nn.Linear(128*32, 256)
        self.lidar4_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar4_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar4_fc1 = nn.Linear(128*32, 256)
        self.lidar5_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar5_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar5_fc1 = nn.Linear(128*32, 256)
        '''


        # Q2 architecture
        '''
        self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256*5, hidden_dim)  # 5 CCTVs
        '''
        self.linear4 = nn.Linear(256 + num_goal_obs + num_vel_obs + num_actions + 256, hidden_dim)  # 1 CCTV
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    #def forward(self, frame, goal, vel, action):
    def forward(self, frame, goal, vel, action, lidar1, lidar2, lidar3, lidar4, lidar5):  # 220713
        o1 = F.relu(self.fea_cv1(frame))
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)
        o1 = F.relu(self.fc1(o1))
        
        # 220812 lidar preprocessing
        l1 = F.relu(self.lidar1_cv1(lidar1))   
        l1 = F.relu(self.lidar1_cv2(l1))
        l1 = l1.view(l1.shape[0], -1)   # (1, 4096)
        l1 = F.relu(self.lidar1_fc1(l1))   # (1, 256)
        '''
        l2 = F.relu(self.lidar2_cv1(lidar2))   
        l2 = F.relu(self.lidar2_cv2(l2))
        l2 = l2.view(l2.shape[0], -1)   # (1, 4096)
        l2 = F.relu(self.lidar2_fc1(l2))   # (1, 256)
        l3 = F.relu(self.lidar3_cv1(lidar3))   
        l3 = F.relu(self.lidar3_cv2(l3))
        l3 = l3.view(l3.shape[0], -1)   # (1, 4096)
        l3 = F.relu(self.lidar3_fc1(l3))   # (1, 256)
        l4 = F.relu(self.lidar4_cv1(lidar4))   
        l4 = F.relu(self.lidar4_cv2(l4))
        l4 = l4.view(l4.shape[0], -1)   # (1, 4096)
        l4 = F.relu(self.lidar4_fc1(l4))   # (1, 256)
        l5 = F.relu(self.lidar5_cv1(lidar5))   
        l5 = F.relu(self.lidar5_cv2(l5))
        l5 = l5.view(l5.shape[0], -1)   # (1, 4096)
        l5 = F.relu(self.lidar5_fc1(l5))   # (1, 256)
        '''
        
        '''
        xu = torch.cat((o1, goal, vel, action, l1, l2, l3, l4, l5), dim=-1) # 5 CCTVs  220813
        '''
        xu = torch.cat((o1, goal, vel, action, l1), dim=-1) # 1 CCTVs 220816
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x2

class GaussianPolicy_CCTV(nn.Module):
    def __init__(self, num_frame_obs, num_goal_obs, num_vel_obs, num_actions, hidden_dim, action_space=None):
                       #    3              2            2          
        super(GaussianPolicy_CCTV, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(2))

        self.fea_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32, 256) # Lidar conv, fc output = 256. it will be next observation full connected layer including goal and velocity
        
        # 220812 lidar
        self.lidar1_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar1_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar1_fc1 = nn.Linear(128*32, 256)
        '''
        self.lidar2_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar2_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar2_fc1 = nn.Linear(128*32, 256)
        self.lidar3_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar3_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar3_fc1 = nn.Linear(128*32, 256)
        self.lidar4_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar4_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar4_fc1 = nn.Linear(128*32, 256)
        self.lidar5_cv1 = nn.Conv1d(in_channels=num_frame_obs, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.lidar5_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.lidar5_fc1 = nn.Linear(128*32, 256)
        '''
        
        #self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs, hidden_dim)
        '''
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + 256*5, hidden_dim)
        '''
        self.linear1 = nn.Linear(256 + num_goal_obs + num_vel_obs + 256, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions) # Different from PPO

        self.mean1_linear = nn.Linear(hidden_dim, 1) # Different from PPO
        self.mean2_linear = nn.Linear(hidden_dim, 1) # Different from PPO

        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            
            scale = [0.5, 1]
            bias = [0.5, 0]

            self.action_scale = torch.FloatTensor(scale)
            self.action_bias = torch.FloatTensor(bias)

            print("self.action_scale: ", self.action_scale)
            print("self.action_bias: ", self.action_bias)

    #def forward(self, frame, goal, vel):
    def forward(self, frame, goal, vel, lidar1, lidar2, lidar3, lidar4, lidar5):  # 220712
        o1 = F.relu(self.fea_cv1(frame))   
        o1 = F.relu(self.fea_cv2(o1))
        o1 = o1.view(o1.shape[0], -1)   # (1, 4096)
        o1 = F.relu(self.fc1(o1))   # (1, 256)
        
        # 220812 lidar preprocessing
        l1 = F.relu(self.lidar1_cv1(lidar1))   
        l1 = F.relu(self.lidar1_cv2(l1))
        l1 = l1.view(l1.shape[0], -1)   # (1, 4096)
        l1 = F.relu(self.lidar1_fc1(l1))   # (1, 256)
        '''
        l2 = F.relu(self.lidar2_cv1(lidar2))   
        l2 = F.relu(self.lidar2_cv2(l2))
        l2 = l2.view(l2.shape[0], -1)   # (1, 4096)
        l2 = F.relu(self.lidar2_fc1(l2))   # (1, 256)
        l3 = F.relu(self.lidar3_cv1(lidar3))   
        l3 = F.relu(self.lidar3_cv2(l3))
        l3 = l3.view(l3.shape[0], -1)   # (1, 4096)
        l3 = F.relu(self.lidar3_fc1(l3))   # (1, 256)
        l4 = F.relu(self.lidar4_cv1(lidar4))   
        l4 = F.relu(self.lidar4_cv2(l4))
        l4 = l4.view(l4.shape[0], -1)   # (1, 4096)
        l4 = F.relu(self.lidar4_fc1(l4))   # (1, 256)
        l5 = F.relu(self.lidar5_cv1(lidar5))   
        l5 = F.relu(self.lidar5_cv2(l5))
        l5 = l5.view(l5.shape[0], -1)   # (1, 4096)
        l5 = F.relu(self.lidar5_fc1(l5))   # (1, 256)
        '''

        # concat lidar_frame, local_goal, velocity
        '''
        state = torch.cat((o1, goal, vel, l1, l2, l3, l4, l5), dim=-1) # 5 CCTVs 220813
        '''
        state = torch.cat((o1, goal, vel, l1), dim=-1) # 5 CCTVs 220813

        x = F.relu(self.linear1(state))   # (1, 256)
        x = F.relu(self.linear2(x))   # (1, 256)
        #mean1 = F.sigmoid(self.mean1_linear(x))
        #mean2 = F.tanh(self.mean2_linear(x))
        
        #mean = torch.cat((mean1, mean2), dim=-1)

        mean = self.mean_linear(x)   # (1, 2)

        #log_std = self.logstd.expand_as(mean)

        log_std = self.log_std_linear(x)   # (1, 2)
        
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, frame, goal, vel, lidar1, lidar2, lidar3, lidar4, lidar5):   # 220712
        '''
        Input: frame_list, goal_list, vel_list
        Output: action(training), log_prob, mean(evaluate)
        '''
        mean, log_std = self.forward(frame, goal, vel, lidar1, lidar2, lidar3, lidar4, lidar5)   # 220712 여기에 로컬 맵 input
        std = log_std.exp()   # e^log_std
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)


        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale  + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_CCTV, self).to(device)


if __name__ == '__main__':
    from torch.autograd import Variable
