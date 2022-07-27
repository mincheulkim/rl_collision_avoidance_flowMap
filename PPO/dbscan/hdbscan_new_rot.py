import copy
import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN


# -*- coding: utf-8 -*-
"""
Project Code: based DBSCAN v1.1, new HDBSCAN
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-07
Contact Info: :
    https://deep-eye.tistory.com
    https://deep-i.net
    
from: https://github.com/DEEPI-LAB/dbscan-python

|--------------------------------------------------------------------------------------|
| Python implementation of 'HDBSCAN' Algorithm                                          |
|                                                |  
|                                                                                      |
|     Inputs:                                                                          |
|         x - A matrix whose columns are feature vectors                               |   입력 X
|                                                                                      |
|     Outputs:                                                                         |
|         An array with either a cluster id number and noise id number for each        |
|         column vector                                                                |
|______________________________________________________________________________________|

"""

import numpy as np
from matplotlib import pyplot as plt

class HDBSCAN_new_rot(object):

    def __init__(self,x, y):   # (입력)x: visible_ped_pose_list  y: visible_ped_poly_vel_list of humans in sensor range
        # The number of input dataset
        self.n = len(x)

        # Cluster
        self.idx = np.full((self.n),0)   # 클러스터링 이전 모든 agent의 초기 index는 0으로 초기화
        self.C = 0
        self.input = x

    def grouping(self, position_array, velocity_array):
        # threshold hyperparameters
        pos = 2.0
        #pos = 1.5
        ori = 45
        #ori = 60
        #vel = 1.0
        vel = 1.5
        params = {'position_threshold': pos,
                    'orientation_threshold': ori / 180.0 * np.pi,
                    'velocity_threshold': vel,
                    #'velocity_ignore_threshold': 0.3}
                    #'velocity_ignore_threshold': 0.05}
                    'velocity_ignore_threshold': 0.05}
        #print('로 벨로시티 어레이:',velocity_array)

        num_people = len(position_array)
        vel_orientation_array = []
        vel_magnitude_array = []
        #print('벨 어레이:',velocity_array)
        #print('llllllllllllllllllllllll')
        for [vx, vy] in velocity_array:
            velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
            if velocity_magnitude < params['velocity_ignore_threshold']:
                # if too slow, then treated as being stationary
                vel_orientation_array.append((0.0, 0.0))
                vel_magnitude_array.append((0.0, 0.0))
            else:
                #vel_orientation_array.append((vx / velocity_magnitude, vy / velocity_magnitude))
                vel_orientation_array.append((vx, vy))
                vel_magnitude_array.append((0.0, velocity_magnitude)) # Add 0 to fool DBSCAN
        # grouping in current frame (three passes, each on different criteria)
        labels = [0] * num_people
        
        
        # First, orientation
        properties = vel_orientation_array
        standard = params['orientation_threshold']
        #print('프로퍼티:',properties)
        #print('오리엔테시션 스탠다드: ',standard)
        max_lb = max(labels)
        for lb in range(max_lb + 1):
            sub_properties = []
            sub_idxes = []
            # Only perform DBSCAN within groups (i.e. have the same membership id)
            for i in range(len(labels)):
                if labels[i] == lb:
                    sub_properties.append(properties[i])
                    sub_idxes.append(i)
            #print('heading 인풋:',sub_properties)
            # If there's only 1 person then no need to further group
            if len(sub_idxes) > 1:
                db = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db.fit_predict(sub_properties)
                ##db1 = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=True, min_samples=1)
                #db1 = hdbscan.HDBSCAN(min_cluster_size=2)
                #db1.fit(sub_properties)
                #sub_labels = db1.labels_
                
                max_label = max(labels)
                
                #print('[DEBUG]서브 레이블:',sub_labels)
                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        #print('[DEBUG]orientation_labels:',labels)
        
        
        # Second, velocity magnitude
        properties = vel_magnitude_array
        #print('벨 메그니튜드 인풋:',properties)
        standard = params['velocity_threshold']
        max_lb = max(labels)
        for lb in range(max_lb + 1):
            sub_properties = []
            sub_idxes = []
            # Only perform DBSCAN within groups (i.e. have the same membership id)
            for i in range(len(labels)):
                if labels[i] == lb:
                    sub_properties.append(properties[i])
                    sub_idxes.append(i)
        
            # If there's only 1 person then no need to further group
            if len(sub_idxes) > 1:
                db2 = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db2.fit_predict(sub_properties)
                ##db2 = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=True, min_samples=1)
                #db2 = hdbscan.HDBSCAN(min_cluster_size=2)
                #db2.fit(sub_properties)
                #sub_labels = db2.labels_
                
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
                        
        #print('[DEBUG]magnitude_labels:',labels)
        
        
        # Final, position
        properties = position_array
        #print('포지션 인풋:',properties)
        standard = params['position_threshold']
        max_lb = max(labels)
        for lb in range(max_lb + 1):
            sub_properties = []
            sub_idxes = []
            # Only perform DBSCAN within groups (i.e. have the same membership id)
            for i in range(len(labels)):
                if labels[i] == lb:
                    sub_properties.append(properties[i])
                    sub_idxes.append(i)
        
            # If there's only 1 person then no need to further group
            if len(sub_idxes) > 1:
                db3 = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db3.fit_predict(sub_properties)
                
                ##db3 = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=True, min_samples=1)
                #db3 = hdbscan.HDBSCAN(min_cluster_size=2)
                #db3.fit(sub_properties)
                #sub_labels = db3.labels_
                
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        
        #print('[DEBUG]position_labels:',labels)
        return labels