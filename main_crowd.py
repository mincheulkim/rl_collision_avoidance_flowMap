import argparse
from random import randrange
import rospy
import numpy as np


from envs import Env, human


class CrowdSimulator:
    def __init__(self):
        self.env = None


    def generate_pos(self, configs, index):
        """
        Generate a position of the i-th human 
        """
        x, y = 0, 0

        max_x = configs.size_x
        max_y = configs.size_y
        
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        
        while True:
            x = np.random.random() * max_x * 0.5 * sign
            y = (np.random.random() - 0.5) * max_y
            collide = False
            for human in env.human_list:
                if norm((x - human.x, y - human.y)) < configs.human.radius * 2:
                    collide = True
                    break
            if not collide:
                break

        return x, y


    def generate_goal_list(self, configs, index):
        """
        Generate a goal list of the i-th human 
        """
        #self.goal_num = configs.goal_num

        self.goal_list = configs.goal_list
        init_goal_index = np.random.rand(0, 10)   # 0 ~ 9
        goal_num = np.random.rand(0, 4)   # max 3 relay goal

        init_goal_list = []
        for i in range(goal_num):
            next_goal = self.goal_list[init_goal_index+i+1]
            init_goal_index.append(next_goal)

        goal_list = init_goal_list

        return goal_list

    def check_all_sub_done(self, human_num):
        for i in range(human_num):
            while self.env.object_state_sub_list[i] is None \
                or self.env.odom_sub_list[i] is None \
                or self.env.check_crash_list[i] is None:
                pass

        rospy.sleep(1.)

    def main(self, configs):
        self.env = Env(configs)
        
        for index in range(configs['human_num']):
            self.env.init_pub(index)
            self.env.init_sub(index)
            pos = self.generate_pos(index)
            self.env.generate_human(index, pos)
            goal_list = self.generate_goal_list(index)
            self.env.set_goal(index, goal_list)

        self.check_all_sub_done()

        sim_timer = rospy.Timer(rospy.Duration(configs['human_control_freq']), self.simulate)

        rospy.on_shutdown(self.shutdown)
        rospy.spin()

    def shutdown(self):
        print('Shutdown crowd simulation.')

    def simulate(self):
        self.env.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs')
    args = parser.parse_args()

    rospy.init_node('crowd_node')

    configs = []

    sim = CrowdSimulator(configs)
    sim.main()