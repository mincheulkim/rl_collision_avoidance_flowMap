
from envs.utils import *
from envs.agent import Agent



class Human(Agent):
    def __init__(self):
        #super().__init__()              # for python>3.0
        super(Agent, self).__init__()    # for python=2.7

        self.cur_goal_index = 0
        self.goal_list = []
        self.goal_num = len(self.goal_list)

    def get_goal_num(self):
        return self.goal_num

    def get_local_goal(self):
        return self.goal_list[self.cur_goal_index]

    def set_goal_list(self, goal_list):
        self.goal_list = goal_list
    
    def check_goal_arrival(self):
        dist2goal = l2distance(self.pos, self.goal_list[self.cur_goal_index])
        if dist2goal < self.configs['goal_boundary']:
            self.change_goal_index(self.cur_goal_index + 1)

    def change_goal_index(self, new_goal_index):
        if new_goal_index >= self.get_goal_num():
            new_goal_index = 0
        self.cur_goal_index = new_goal_index