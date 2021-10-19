import argparse
from random import randrange
import rospy


from envs import Env, human


def main(configs):
    env = Env(configs)
    human_num = configs['human_num']

    for i in range(human_num):
        env.init_pub()
        env.init_sub()


    env.generate_humans(human_num)

    for i in range(human_num):
        env.set_goal(i, goal_list)

    while True:
        env.step()
        rospy.Rate(configs['rate'])

    # rospy.on_shutdown(nn_jackal.on_shutdown)
    # rospy.spin()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs')
    args = parser.parse_args()

    rospy.init_node('crowd_node')

    configs = []
    main(configs)
