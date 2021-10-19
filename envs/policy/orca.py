import rvo2
import cv2
import numpy as np


class ORCA:
    def __init__(self, configs):
        self.rvo_sim = rvo2.PyRVOSimulator(
            configs['dt'], configs['neighbor_dist'],
            configs['max_neighbors'], configs['time_horizon'],
            configs['time_horizon_obs'], configs['human_radius'],
            configs['max_speed'], (configs['linear_vel'], configs['angular_vel']))

    def add_static_obstacle(self, map_img_path):
        map_img = cv2.imread(map_img_path)
        # Get contours from image to distinguish obstacles
        imgray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(imgray, 0.3, 1, cv2.THRESH_BINARY)
        thr = thr.astype(np.uint8)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add obstacles to sim.
        obs_list = []
        for line_seg in contours:
            obs_boundary = []
            for point in line_seg:
                x, y = point[0]
                obs_boundary.append((x, y))
            obs_list.append(obs_boundary)
        for obs in obs_list:
            self.rvo_sim.addObstacle(obs)
        self.rvo_sim.processObstacles()


if __name__ == '__main__':
    rvo_sim = ORCA()
    rvo_sim.doStep()

    for s in range(100):
        rvo_sim.doStep()
