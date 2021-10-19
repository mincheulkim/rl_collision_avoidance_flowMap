import rospy
import tf
import numpy as np

from envs.policy import ORCA
from envs.human import Human

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int8


class Env:
    def __init__(self, configs):
        self.configs = configs
        self.human_list = []
        self.human_index_list = []
        self.rvo_agent_list = []

        self.sim = ORCA(configs)

        map_img_path = configs['map_img_path']
        self.sim.add_static_obstacle(map_img_path)

        # pub list
        self.cmd_vel_list = []
        self.cmd_pose_list = []

        # sub list
        self.object_state_sub_list = []
        self.odom_sub_list = []
        self.check_crash_list = []
        
        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

    def generate_human(self, index, pos):
        human = Human(pos)
        self.human_list.append(human)
        self.human_index_list.append(index)
        agent = self.sim.addAgent(pos[0], pos[1])
        self.rvo_agent_list.append(agent)

    def init_pub(self, index):
        cmd_vel_topic = 'human_' + str(index) + '/cmd_vel'
        cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.cmd_vel_list.append(cmd_vel)        
        
        cmd_pose_topic = 'human_' + str(index) + '/cmd_pose'
        cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)
        self.cmd_vel_list.append(cmd_pose)        


    def init_sub(self, index):
        object_state_topic = 'human_' + str(index) + '/base_pose_ground_truth'
        object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)
        self.object_state_sub_list.append(object_state_sub)
        
        odom_topic = 'human_' + str(index) + '/odom'
        odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)
        self.odom_sub_list.append(odom_sub)
        
        crash_topic = 'human_' + str(index) + '/is_crashed'
        check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)
        self.check_crash_list.append(check_crash)

    def step(self):
        self.sim.doStep()
        for index in self.human_index_list:
            human = self.human_list[index]
            cur_pos = human.get_pos()
            self.sim.setAgentPosition(cur_pos)
            goal_pos = human.get_local_goal()
            dt_vel = (goal_pos[0] - cur_pos[0], goal_pos[1] - cur_pos[1])
            dt_angle = self.compute_rel_angle(goal_pos, cur_pos)
            agent = self.rvo_agent_list[index]
            self.sim.setAgentPrefVelocity(agent, dt_vel)
            pref_linear_vel, pref_angular_vel = human.get_linear_vel[1], human.get_angular_vel[1]
            self.control_vel([pref_linear_vel, pref_angular_vel * dt_angle])

    def set_goal(self, index, goal_list):
        human = self.human_list[index]
        human.set_goal_list(goal_list)

    def compute_rel_angle(self, start_point, end_point):
        rel_angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        return rel_angle

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def reset_pose(self):
        random_pose = self.generate_random_pose()   # return [x, y, theta]   [-9~9,-9~9], dist>9
        rospy.sleep(0.01)
        self.control_pose(random_pose)   # create pose(Euler or quartanion) for ROS
        [x_robot, y_robot, theta] = self.get_self_stateGT()   # Ground Truth Pose

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:  # np.bas: absolute, compare # generated random pose with topic pose
            [x_robot, y_robot, theta] = self.get_self_stateGT()    # same
            self.control_pose(random_pose)
        rospy.sleep(0.01)


    def control_vel(self, action, index):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel_list[index].publish(move_cmd)


    def control_pose(self, pose, index):    # pose = [x, y, theta]
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]   # x
        pose_cmd.position.y = pose[1]   # y
        pose_cmd.position.z = 0         # 0(don't care rot cause pose?)

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose_list[index].publish(pose_cmd)

#############################################################################################
#                                   Callback function
#############################################################################################
    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data




