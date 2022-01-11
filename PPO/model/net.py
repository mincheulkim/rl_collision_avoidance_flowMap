import math
import numpy as np
import torch
import torch.nn as nn

import cv2
from torch.nn import init
from torch.nn import functional as F


from model.utils import log_normal_density

class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)


class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):
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

        # print(x.shape)
        # print(goal.shape)
        # print(speed.shape)

        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        #mean1 = F.sigmoid(self.actor1(a))
        #mean2 = F.tanh(self.actor2(a))
        mean1 = torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        
        #---------------------------------------------------------------------#

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
    
    
    
    
    
# 220105    
class LM_Policy(nn.Module):
    def __init__(self, frames, action_space):
        super(LM_Policy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        #self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.act_fc2 =  nn.Linear(256+2+2+512, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        #self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.crt_fc2 = nn.Linear(256+2+2+512, 128)
        self.critic = nn.Linear(128, 1)
        
        
        
        # convLSTM
        nf = 3
        input_chanel = 3
        padding = 3 // 2, 3 // 2
        self.act_conv = nn.Conv2d(in_channels=input_chanel+nf, out_channels=4*nf, kernel_size=(3,3), padding=padding, bias=True)
        #self.conv2 = nn.Conv2d(in_channels=nf+nf, out_channels=4*nf, kernel_size=(3,3), padding=padding, bias=True)
        self.crt_conv = nn.Conv2d(in_channels=input_chanel+nf, out_channels=4*nf, kernel_size=(3,3), padding=padding, bias=True)
        
        self.act_conv_fc = nn.Linear(nf*15*15, 512)
        self.crt_conv_fc = nn.Linear(nf*15*15, 512)
        
        

    def forward(self, x, goal, speed, local_maps):
        """
            returns value estimation, action, log_action_prob
        """
         # action
        #print(x.shape)      # 1, 3, 512                                    # 1024, 3, 512  
        #print(goal.shape)   # 1, 2                                         # 1024, 2
        #print(speed.shape)  # 1, 2                                         # 1024, 2
        #print(local_maps.shape)  # 1, 8, 3, 60, 60    B, S, C, W, H        # 1024, 8, 3, 60, 60

        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a)) # 1, 32, 128
        a = a.view(a.shape[0], -1) #1, 4096
        a = F.relu(self.act_fc1(a))
        
        # convLSTM
        # initialize hidden
        h_t, c_t = torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.act_conv.weight.device),torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.act_conv.weight.device)
        #h_t2, c_t2 = torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device), torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device)
        
        for t in range(local_maps.shape[1]):   # 8
            
            input_tensor = local_maps[:, t, :, :]
            cur_state=[h_t, c_t]
            
            h_cur, c_cur = cur_state

            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

            combined_conv = self.act_conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 3, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            h_t=h_next
            c_t=c_next
            
        # encoder_vector
        encoder_vector = h_t
        
        '''
        # 220110 Visualize Feature map (three channel)
        show_h_t= encoder_vector[0, :, :, :]
        show_h_t=show_h_t.transpose(0,1)
        show_h_t=show_h_t.transpose(1,2)
        show_h_t=show_h_t.cpu().detach().numpy()

        hsv = cv2.cvtColor(np.float32(show_h_t), cv2.COLOR_RGB2HSV)
        hsv=cv2.resize(hsv, dsize=(240,240), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image',hsv)
        cv2.waitKey(1)
        '''
        
        encoder_vector=F.max_pool2d(encoder_vector, 2)                          
        encoder_vector=F.max_pool2d(encoder_vector, 2)              # 1, 3, 15, 15              

        '''
        # 220110 Visualize Feature map (one channel)
        show_h_t=encoder_vector[:,9,:,:]   # 1, 3, 15, 15
        show_h_t=show_h_t.transpose(0, 1)  # 15, 3, 15
        show_h_t=show_h_t.transpose(1, 2)  # 15, 15, 3
        show_h_t=show_h_t.cpu().detach().numpy()
        dist = cv2.resize(show_h_t*10, dsize=(480,480), interpolation=cv2.INTER_LINEAR)   # https://076923.github.io/posts/Python-opencv-8/
        cv2.imshow("Local flow map", dist)
        cv2.waitKey(1)
        '''
        
        encoder_vector=encoder_vector.view(encoder_vector.shape[0], -1) #1, 675
        vvv=F.relu(self.act_conv_fc(encoder_vector))
        #print(vvv.shape)    # 1, 512
        
        # dbscan 그룹 채널을 convlstm 채널에 넣기
        #a = torch.cat((a, goal, speed), dim=-1)
        a = torch.cat((a, goal, speed, vvv), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        
        #---------------------------------------------------------------------#
        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        
        
        
       # convLSTM
        # initialize hidden
        h_t_c, c_t_c = torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.crt_conv.weight.device),torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.crt_conv.weight.device)
        #h_t2, c_t2 = torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device), torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device)
        
        for t in range(local_maps.shape[1]):   # 8
            
            input_tensor = local_maps[:, t, :, :]
            cur_state=[h_t_c, c_t_c]
            
            h_cur, c_cur = cur_state

            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

            combined_conv = self.crt_conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 3, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            h_t_c=h_next
            c_t_c=c_next
            
        # encoder_vector
        #encoder_vector = h_t2
        encoder_vector_c = h_t_c
        encoder_vector_c=F.max_pool2d(encoder_vector_c, 2)                          
        encoder_vector_c=F.max_pool2d(encoder_vector_c, 2)                          
        #print(x.shape, encoder_vector.shape)   # 1, 16, 15, 15  
        encoder_vector_c=encoder_vector_c.view(encoder_vector_c.shape[0], -1) #1, 3600
        vvvv=F.relu(self.crt_conv_fc(encoder_vector_c))
                
        v = torch.cat((v, goal, speed, vvvv), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action, local_maps):
        v, _, _, mean = self.forward(x, goal, speed, local_maps)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy
    
    
    
    
    
    
    

# 211213
class stacked_LM_Policy(nn.Module):
    def __init__(self, frames, action_space):
        super(stacked_LM_Policy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2+256, 512)
        self.act_fc3 = nn.Linear(512, 256)
        self.actor1 = nn.Linear(256, 1)
        self.actor2 = nn.Linear(256, 1)
        
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2+256, 512)
        self.crt_fc3 = nn.Linear(512, 256)
        self.critic = nn.Linear(256, 1)

        '''
        # For Local maps  if kernel_size=3, padding =1: output is same as input size
        self.act_fea_LM_cv1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1)   # kernel size = filter size()
        self.act_fea_LM_cv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1)
        
        # Maxpool 2D
        self.crt_fea_LM_cv1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1)
        self.crt_fea_LM_cv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1)
        '''
        
        # Maxpool 2D
        self.C_in = 10
        # 220103 Depth-wise convolution   https://supermemi.tistory.com/118
        self.op_act = nn.Sequential(nn.Conv2d(self.C_in, self.C_in, kernel_size=3, padding=1, groups=self.C_in, bias=False),
                  nn.BatchNorm2d(self.C_in),
                  nn.ReLU(inplace=False),
                  nn.MaxPool2d(2),
                  nn.Conv2d(self.C_in, self.C_in, kernel_size=3, padding=1, groups=self.C_in, bias=False),
                  nn.BatchNorm2d(self.C_in),
                  nn.ReLU(inplace=False),
                  nn.MaxPool2d(2)
                  )
        self.act_LM_fc1 = nn.Linear(15*15*self.C_in, 256)

        self.op_crt = nn.Sequential(nn.Conv2d(self.C_in, self.C_in, kernel_size=3, padding=1, groups=self.C_in, bias=False),
                  nn.BatchNorm2d(self.C_in),
                  nn.ReLU(inplace=False),
                  nn.MaxPool2d(2),
                  nn.Conv2d(self.C_in, self.C_in, kernel_size=3, padding=1, groups=self.C_in, bias=False),
                  nn.BatchNorm2d(self.C_in),
                  nn.ReLU(inplace=False),
                  nn.MaxPool2d(2)
                  )
        self.crt_LM_fc1 = nn.Linear(15*15*self.C_in, 256)


    def forward(self, x, goal, speed, local_maps):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        #print(x.shape)      # 1, 3, 512                 # 1024, 3, 512  (batch)
        #print(goal.shape)   # 1, 2                      # 1024, 2
        #print(speed.shape)  # 1, 2                      # 1024, 2
        #print(local_maps.shape)  # 1, 3(5), 60, 60         # 1024, 3(5), 60, 60

        a = F.relu(self.act_fea_cv1(x))      # 1, 32, 255    # LIDAR input shape: [# of examples(batchsize), timesteps, feature)]
        a = F.relu(self.act_fea_cv2(a))      # 1, 32, 128
        a = a.view(a.shape[0], -1)     # a.shape[0]=1, so (1, 4096)
        a = F.relu(self.act_fc1(a))          # 1, 256
        
        
        # Local maps embedding
        '''
        b = F.relu(self.act_fea_LM_cv1(local_maps))    # (1, 32, 60, 60)
        b=F.max_pool2d(b, 2)                           # (1, 32, 30, 30)
        b = F.relu(self.act_fea_LM_cv2(b))             # (1, 32, 30, 30)
        b=F.max_pool2d(b, 2)                           # (1, 32, 15, 15)
        b=b.view(b.shape[0], -1)                       # (1, 7200)
        b=F.relu(self.act_LM_fc1(b))                   # 1, 256
        '''
        
        # Depth-wise convolution (each channel saparate values)
        austion = self.op_act(local_maps)
        # TODO: attention
        austion = austion.view(austion.shape[0], -1)
        
        b = F.relu(self.act_LM_fc1(austion))
        #torch.set_printoptions(threshold=10_000)
        #print(austion.view(austion.shape[0],austion.shape[1],-1))
    

        a = torch.cat((a, goal, speed, b), dim=-1)   # added b   new 211214
        a = F.relu(self.act_fc2(a))
        a = F.relu(self.act_fc3(a))                  # new 211214
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        
        #---------------------------------------------------------------------#

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))

        '''
        b_c = F.relu(self.crt_fea_LM_cv1(local_maps))    # (1, 32, 60, 60)
        b_c=F.max_pool2d(b_c, 2)                           # (1, 32, 30, 30)
        b_c = F.relu(self.crt_fea_LM_cv2(b_c))             # (1, 32, 30, 30)
        b_c=F.max_pool2d(b_c, 2)                           # (1, 32, 15, 15)
        b_c=b_c.view(b_c.shape[0], -1)                       # (1, 7200)
        b_c=F.relu(self.crt_LM_fc1(b_c))                   # 1, 256
        '''
        
        # Depth-wise convolution (each channel saparate values)
        austion_crt = self.op_crt(local_maps)
        # TODO: attention
        austion_crt = austion_crt.view(austion_crt.shape[0], -1)
        b_c = F.relu(self.act_LM_fc1(austion_crt))
        #torch.set_printoptions(threshold=10_000)
        #print(austion.view(austion.shape[0],austion.shape[1],-1))

        v = torch.cat((v, goal, speed, b_c), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = F.relu(self.crt_fc3(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action, local_maps):
        v, _, _, mean = self.forward(x, goal, speed, local_maps)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy








# 220105    
class concat_LM_Policy(nn.Module):
    def __init__(self, frames, action_space):
        super(concat_LM_Policy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        #self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.act_fc2 =  nn.Linear(256+2+2+512, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)
        # For discrete action
        #self.discrete = nn.Linear(128, 35)

        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        #self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.crt_fc2 = nn.Linear(256+2+2+512, 128)
        self.critic = nn.Linear(128, 1)
        
        
        
        # convLSTM
        nf = 3
        input_chanel = 3
        padding = 3 // 2, 3 // 2
        self.act_conv = nn.Conv2d(in_channels=input_chanel+nf, out_channels=4*nf, kernel_size=(3,3), padding=padding, bias=True)
        #self.conv2 = nn.Conv2d(in_channels=nf+nf, out_channels=4*nf, kernel_size=(3,3), padding=padding, bias=True)
        self.crt_conv = nn.Conv2d(in_channels=input_chanel+nf, out_channels=4*nf, kernel_size=(3,3), padding=padding, bias=True)
        
        self.act_conv_fc = nn.Linear(nf*15*15, 512)
        self.crt_conv_fc = nn.Linear(nf*15*15, 512)
        
        

    def forward(self, x, goal, speed, local_maps):
        """
            returns value estimation, action, log_action_prob
        """
         # action
        #print(x.shape)      # 1, 3, 512                                    # 1024, 3, 512  
        #print(goal.shape)   # 1, 2                                         # 1024, 2
        #print(speed.shape)  # 1, 2                                         # 1024, 2
        #print(local_maps.shape)  # 1, 8, 3, 60, 60    B, S, C, W, H        # 1024, 8, 3, 60, 60

        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a)) # 1, 32, 128
        a = a.view(a.shape[0], -1) #1, 4096
        a = F.relu(self.act_fc1(a))
        
        # convLSTM
        # initialize hidden
        h_t, c_t = torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.act_conv.weight.device),torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.act_conv.weight.device)
        #h_t2, c_t2 = torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device), torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device)
        
        for t in range(local_maps.shape[1]):   # 8
            
            input_tensor = local_maps[:, t, :, :]
            cur_state=[h_t, c_t]
            
            h_cur, c_cur = cur_state

            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

            combined_conv = self.act_conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 3, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            h_t=h_next
            c_t=c_next
            
        # encoder_vector
        encoder_vector = h_t
        
        '''
        # 220110 Visualize Feature map (three channel)
        show_h_t= encoder_vector[0, :, :, :]
        show_h_t=show_h_t.transpose(0,1)
        show_h_t=show_h_t.transpose(1,2)
        show_h_t=show_h_t.cpu().detach().numpy()

        hsv = cv2.cvtColor(np.float32(show_h_t), cv2.COLOR_RGB2HSV)
        hsv=cv2.resize(hsv, dsize=(240,240), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image',hsv)
        cv2.waitKey(1)
        '''
        
        encoder_vector=F.max_pool2d(encoder_vector, 2)                          
        encoder_vector=F.max_pool2d(encoder_vector, 2)              # 1, 3, 15, 15              

        '''
        # 220110 Visualize Feature map (one channel)
        show_h_t=encoder_vector[:,9,:,:]   # 1, 3, 15, 15
        show_h_t=show_h_t.transpose(0, 1)  # 15, 3, 15
        show_h_t=show_h_t.transpose(1, 2)  # 15, 15, 3
        show_h_t=show_h_t.cpu().detach().numpy()
        dist = cv2.resize(show_h_t*10, dsize=(480,480), interpolation=cv2.INTER_LINEAR)   # https://076923.github.io/posts/Python-opencv-8/
        cv2.imshow("Local flow map", dist)
        cv2.waitKey(1)
        '''
        
        encoder_vector=encoder_vector.view(encoder_vector.shape[0], -1) #1, 675
        vvv=F.relu(self.act_conv_fc(encoder_vector))
        #print(vvv.shape)    # 1, 512
        
        # dbscan 그룹 채널을 convlstm 채널에 넣기
        #a = torch.cat((a, goal, speed), dim=-1)
        a = torch.cat((a, goal, speed, vvv), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = torch.sigmoid(self.actor1(a))   # 0~1, linear vel
        mean2 = torch.tanh(self.actor2(a))      # -1~1, angular rot
        mean = torch.cat((mean1, mean2), dim=-1)
        #print('mean:',mean)
        logstd = self.logstd.expand_as(mean)    # mean처럼 [2,] 즉 [[0,0]]으로 확장한다
        std = torch.exp(logstd)     # [[1,1]]
        action = torch.normal(mean, std)
        #print(action)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        
        # TODO discrete action:
        # 1. 512 -> 28 FC layer추가
        # 2. 28 FC -> softmax layer 거치게 함. 이 값은 28개쌍의 (linear, angular) 쌍에 할당. linear:{0.0, 0.2, 0.4, 0.6}
        linear_vel = [0.0, 0.25, 0.50, 0.75, 1.0]  # 5
        angular_vel = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]  # 7
        
        action_space = [[0.0,-0.9],[0.0,-0.6],[0.0,-0.3],[0.0,0.0],[0.0,0.3],[0.0,0.6],[0.0,0.9],
                        [0.25,-0.9],[0.25,-0.6],[0.25,-0.3],[0.25,0.0],[0.25,0.3],[0.25,0.6],[0.25,0.9],
                        [0.50,-0.9],[0.50,-0.6],[0.50,-0.3],[0.50,0.0],[0.50,0.3],[0.50,0.6],[0.50,0.9],
                        [0.75,-0.9],[0.75,-0.6],[0.75,-0.3],[0.75,0.0],[0.75,0.3],[0.75,0.6],[0.75,0.9],
                        [1.0,-0.9],[1.0,-0.6],[1.0,-0.3],[1.0,0.0],[1.0,0.3],[1.0,0.6],[1.0,0.9]]       # 5 * 7 = 35 discrete actions 장점: no clipping
    
        # TODO continuous action다르게
        # 1. mean 가질때, self.actor1(a), self.actor2(a)로 따로 하지말고 (128, 2)의 self.actor(a)의 2 units 거쳐서 각각 0, 1이 mean1, mean2 되게
        #ppi = softmax(self.discrete(a), dim=1)
        #ppi = Categorical(probs=ppi)
        #a=ppi.sample()
        #a=a.data.item()
        #action = action_space[a]
        
        
        #---------------------------------------------------------------------#
        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        
        
        
        # convLSTM
        # initialize hidden
        h_t_c, c_t_c = torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.crt_conv.weight.device),torch.zeros(local_maps.shape[0], 3, 60, 60, device=self.crt_conv.weight.device)
        #h_t2, c_t2 = torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device), torch.zeros(local_maps.shape[0], 64, 60, 60, device=self.conv2.weight.device)
        
        for t in range(local_maps.shape[1]):   # 8
            
            input_tensor = local_maps[:, t, :, :]
            cur_state=[h_t_c, c_t_c]
            
            h_cur, c_cur = cur_state

            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

            combined_conv = self.crt_conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 3, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            h_t_c=h_next
            c_t_c=c_next
            
        # encoder_vector
        #encoder_vector = h_t2
        encoder_vector_c = h_t_c
        encoder_vector_c=F.max_pool2d(encoder_vector_c, 2)                          
        encoder_vector_c=F.max_pool2d(encoder_vector_c, 2)                          
        #print(x.shape, encoder_vector.shape)   # 1, 16, 15, 15  
        encoder_vector_c=encoder_vector_c.view(encoder_vector_c.shape[0], -1) #1, 3600
        vvvv=F.relu(self.crt_conv_fc(encoder_vector_c))
                
        v = torch.cat((v, goal, speed, vvvv), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action, local_maps):
        v, _, _, mean = self.forward(x, goal, speed, local_maps)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy








class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
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


if __name__ == '__main__':
    from torch.autograd import Variable

    net = MLPPolicy(3, 2)

    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print(v)

