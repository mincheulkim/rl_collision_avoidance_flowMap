import argparse
from random import randrange
import rospy


from envs import Env, human


class CrowdSimulator:
    def __init__(self):
        self.env = None


    def generate_pos(self, configs, index):
        """
        Generate a position of the i-th human 
        """
        x, y = 0, 0

        # TODO
        # Do something
        #

        return x, y


    def generate_goal_list(self, configs, index):
        """
        Generate a goal list of the i-th human 
        """
        goal_list = []

        # TODO
        # Do something
        #

        return goal_list

    def check_all_sub_done(self, human_num):
        for i in range(human_num):
            self.object_state_sub_list = []
            self.odom_sub_list = []
            self.check_crash_list = []

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
            pos = self.generate_pos(index)  # TODO
            self.env.generate_human(index, pos)
            goal_list = self.generate_goal_list(index)  # TODO
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