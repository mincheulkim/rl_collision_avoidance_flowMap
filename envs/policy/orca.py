import rvo2
import cv2
import numpy as np


class ORCA:
    def __init__(self, configs):
        
        self.rvo_sim = rvo2.PyRVOSimulator(
            float(configs['ORCA']['dt']), float(configs['ORCA']['neighbor_dist']),                # https://dojang.io/mod/page/view.php?id=2310
            float(configs['ORCA']['max_neighbors']), float(configs['ORCA']['time_horizon']),
            float(configs['ORCA']['time_horizon_obs']), float(configs['ORCA']['human_radius']),
            float(configs['ORCA']['max_speed']))
        
        #self.sim = rvo2.PyRVOSimulator(float(configs['ORCA']['dt']), 1.5, 5, 1.5, 2, 0.4, 2)


    def add_static_obstacle(self, map_img_path):
        map_img = cv2.imread(map_img_path)
        # Get contours from image to distinguish obstacles
        imgray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(imgray, 0.3, 1, cv2.THRESH_BINARY)
        thr = thr.astype(np.uint8)
        #contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # https://076923.github.io/posts/Python-opencv-21/
        contours = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add obstacles to sim.
        obs_list = []
        for line_seg in contours:
            obs_boundary = []
            #print('line seq:',line_seg)
            '''
            for point in line_seg:
                x, y = point[0]
                obs_boundary.append((x, y))
            '''
            obs_list.append(obs_boundary)
        for obs in obs_list:
            #self.rvo_sim.addObstacle(obs)
            print('need to fix')
        self.rvo_sim.processObstacles()


if __name__ == '__main__':
    rvo_sim = ORCA()
    rvo_sim.doStep()

    for s in range(100):
        rvo_sim.doStep()
