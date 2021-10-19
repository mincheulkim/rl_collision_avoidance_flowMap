
from envs.utils import *
from envs.agent import Agent



class Human(Agent):
    def __init__(self) -> None:
        super().__init__()

        self.cur_goal_index = 0
        self.goal_list = []
    
    def arrive_goal(self):
        goal_dist = l2distance(self.pos, self.goal_list[self.cur_goal_index])
        if goal_dist < self.configs['goal_boundary']:
            self.cur_goal_index += 1

            if self.cur_goal_index >= len(self.goal_list):
                self.cur_goal_index = 0    
    
