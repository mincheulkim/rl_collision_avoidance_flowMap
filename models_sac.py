import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        mu = torch.tanh(self.fc4(h_fc3))
        return mu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
class PolicyNetGaussian(nn.Module):
    def __init__(self):
        super(PolicyNetGaussian, self).__init__()
        '''
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_mean = nn.Linear(512, 2)
        self.fc4_logstd = nn.Linear(512, 2)
        '''
        self.act_fea_cv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1)   # lidar
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)       # lidar
        self.act_fc1 = nn.Linear(128*32, 256)     # 4096->256
        self.act_fc2 =  nn.Linear(256+2+2, 256)   # 260-> 128
        self.fc3 =  nn.Linear(256, 256)   # 260-> 128
        self.fc4_mean = nn.Linear(256, 2)
        self.fc4_logstd = nn.Linear(256, 2)

    def forward(self, state, goal, vel):
        '''
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        a_mean = self.fc4_mean(h_fc3)
        a_logstd = self.fc4_logstd(h_fc3)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        '''
        h_cv1 = F.relu(self.act_fea_cv1(state))   # (1,3,512) -> (1,32,255)
        h_cv2 = F.relu(self.act_fea_cv2(h_cv1))   # (1,3,255) -> (1,32,128)
        h_cv3 = h_cv2.view(h_cv2.shape[0], -1)        # (1,32,128) -> (1,4096)
        h_fc1 = F.relu(self.act_fc1(h_cv3))       # (1,4096) -> (1, 256)
        h_fc2 = torch.cat((h_fc1, goal, vel), dim=-1)
        h_fc3 = F.relu(self.act_fc2(h_fc2))
        h_fc4 = F.relu(self.fc3(h_fc3))
        a_mean = self.fc4_mean(h_fc4)
        a_logstd = self.fc4_logstd(h_fc4)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return a_mean, a_logstd
    
    def sample(self, state, goal, vel):
        a_mean, a_logstd = self.forward(state, goal, vel)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        '''
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512+2, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        '''

        self.act_fea_cv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1)   # lidar
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)       # lidar
        self.act_fc1 = nn.Linear(128*32, 256)     # 4096->256
        self.act_fc2 =  nn.Linear(256+2+2, 256)   # 260-> 128
        self.fc3 =  nn.Linear(256+2, 256)   # 260-> 128
        self.fc4 = nn.Linear(256, 1)
    
    #def forward(self, s, a):
    def forward(self, state, goal, vel, a):
        '''
        h_fc1 = F.relu(self.fc1(s))
        h_fc1_a = torch.cat((h_fc1, a), 1)
        h_fc2 = F.relu(self.fc2(h_fc1_a))
        h_fc3 = F.relu(self.fc3(h_fc2))
        q_out = self.fc4(h_fc3)
        '''

        h_cv1 = F.relu(self.act_fea_cv1(state))   # (1,3,512) -> (1,32,255)
        h_cv2 = F.relu(self.act_fea_cv2(h_cv1))   # (1,3,255) -> (1,32,128)
        h_cv3 = h_cv2.view(h_cv2.shape[0], -1)        # (1,32,128) -> (1,4096)
        h_fc1 = F.relu(self.act_fc1(h_cv3))       # (1,4096) -> (1, 256)
        h_fc2 = torch.cat((h_fc1, goal, vel), dim=-1)
        h_fc3 = F.relu(self.act_fc2(h_fc2))
        h_fc3_a = torch.cat((h_fc3, a), 1)
        h_fc4 = F.relu(self.fc3(h_fc3_a))
        q_out = self.fc4(h_fc4)
        return q_out