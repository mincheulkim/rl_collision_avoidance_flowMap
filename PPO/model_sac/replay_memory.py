import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    #def list_push(self, state_list, action_list, reward_list, next_state_list, done_list):
    #    s_list, goal_list, speed_list = [], [], []

    def push_ped(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, ped, n_ped):
        if len(self.buffer) < self.capacity:   # 220720
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done, ped, n_ped)
        self.position = (self.position + 1) % self.capacity
        
    def push_mask(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, mask, n_mask):
        if len(self.buffer) < self.capacity:   # 220720
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done, mask, n_mask)
        self.position = (self.position + 1) % self.capacity
        
    def push_cctv(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5):  # 220812
        if len(self.buffer) < self.capacity:   # 220720
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5)
        self.position = (self.position + 1) % self.capacity
        
    def push_cctv_header(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5, cctv_header):  # 220812
        if len(self.buffer) < self.capacity:   # 220822
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5, cctv_header)
        self.position = (self.position + 1) % self.capacity
        
    def push_iros(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, ped, n_ped):
        if len(self.buffer) < self.capacity:   # 220720
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done, ped, n_ped)
        self.position = (self.position + 1) % self.capacity
        
    def push(self, frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (frame,goal,speed, action, reward, n_frame, n_goal, n_speed, done)
        self.position = (self.position + 1) % self.capacity

    def sample_ped(self, batch_size):   # 220720
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, ped, n_ped = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame,n_goal, n_speed, done, ped, n_ped
    
    def sample_mask(self, batch_size):   # 220720
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, mask, n_mask = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame,n_goal, n_speed, done, mask, n_mask
    
    def sample_cctv(self, batch_size):   # 220812
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5 = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame,n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5
    
    def sample_cctv_header(self, batch_size):   # 220822
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5, cctv_header = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame,n_goal, n_speed, done, lidar1, n_lidar1, lidar2, n_lidar2,lidar3, n_lidar3,lidar4, n_lidar4,lidar5, n_lidar5, cctv_header
    
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame,n_goal, n_speed, done
    
    def sample_iros(self, batch_size):   # 220720
        batch = random.sample(self.buffer, batch_size)
        frame, goal, speed, action, reward, n_frame, n_goal, n_speed, done, ped, n_ped = map(np.stack, zip(*batch))
        return frame, goal, speed, action, reward, n_frame,n_goal, n_speed, done, ped, n_ped

    def __len__(self):
        return len(self.buffer)