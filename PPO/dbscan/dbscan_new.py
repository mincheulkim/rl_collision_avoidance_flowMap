import copy
import numpy as np
from sklearn.cluster import DBSCAN
# -*- coding: utf-8 -*-
"""
Project Code: DBSCAN v1.1
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-07
Contact Info: :
    https://deep-eye.tistory.com
    https://deep-i.net
    
from: https://github.com/DEEPI-LAB/dbscan-python

|--------------------------------------------------------------------------------------|
| Python implementation of 'DBSCAN' Algorithm                                          |
|     See : https://deep-eye.tistory.com/36                                            |  
|                                                                                      |
|     Inputs:                                                                          |
|         x - A matrix whose columns are feature vectors                               |   입력 X
|         epsilon - The radius of a neighborhood with respect to some point            |   최소 이웃반경 epsilon
|         minPts - The minimum number of points required to form a dense region        |   최소 이웃 개수 minPts
|                                                                                      |
|     Outputs:                                                                         |
|         An array with either a cluster id number and noise id number for each        |
|         column vector                                                                |
|______________________________________________________________________________________|

"""

import numpy as np
from matplotlib import pyplot as plt

class DBSCAN_new(object):

    #def __init__(self,x,epsilon,minpts):   # x: 입력, epsilon: 최소 이웃 반경  minpts: 최소 이우 ㅅ개수
    def __init__(self,x, y,epsilon,minpts):   # x: pose_list  y: poly_speed_list of humans
        # The number of input dataset
        self.n = len(x)
        # Euclidean distance
        p, q = np.meshgrid(np.arange(self.n), np.arange(self.n))   # 격자 그리드 만들기
        self.dist = np.sqrt(np.sum(((x[p] - x[q])**2),2))
        # Relative velocity
        self.dist_vel = np.sqrt(np.sum(((y[p] - y[q])**2),2))
        hyper_params = [5]
        #self.dist = (np.add(self.dist,np.multiply(hyper_params,self.dist_vel)))/(np.add([1],hyper_params))   # option 1 (나누는 숫자가 너무 커서 전반적을 작아짐(관대해짐))
        self.dist = (np.add(self.dist,np.multiply(hyper_params,self.dist_vel)))/2         # option 2   (나누는 숫자 고정)
        self.visited = np.full((self.n), False)     # 방문 flag
        self.noise = np.full((self.n),False)        # 잡음 인덱스  -> 한번 연산된 객체와 잡음 객체는 다시 연산되지 않도록
        # DBSCAN Parameters
        self.epsilon = epsilon
        self.minpts = minpts
        # Cluster
        self.idx = np.full((self.n),0)   # 클러스터링 이전 모든 agent의 초기 index는 0으로 초기화
        self.C = 0
        self.input = x

    def DBScan_grouping(labels, properties, standard):
        # DBSCAN clustering
        # Inputs:
        # labels: the input labels. This will be destructively updated to 
        #         reflect the group memberships after DBSCAN.
        # properties: the input that clustering is based on.
        #             Could be positions, velocities or orientation.
        # standard: the threshold value for clustering.

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
                db = DBSCAN(eps = standard, min_samples = 2)
                sub_labels = db.fit_predict(sub_properties)
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        return labels

    def grouping(self, position_array, velocity_array):
        
        pos = 2.0
        #ori = 45
        ori = 60
        #vel = 1.0
        vel = 1.5
        params = {'position_threshold': pos,
                    'orientation_threshold': ori / 180.0 * np.pi,
                    'velocity_threshold': vel,
                    #'velocity_ignore_threshold': 0.3}
                    'velocity_ignore_threshold': 0.05}

        num_people = len(position_array)
        vel_orientation_array = []
        vel_magnitude_array = []
        for [vx, vy] in velocity_array:
            velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
            if velocity_magnitude < params['velocity_ignore_threshold']:
                # if too slow, then treated as being stationary
                vel_orientation_array.append((0.0, 0.0))
                vel_magnitude_array.append((0.0, 0.0))
            else:
                vel_orientation_array.append((vx / velocity_magnitude, vy / velocity_magnitude))
                vel_magnitude_array.append((0.0, velocity_magnitude)) # Add 0 to fool DBSCAN
        # grouping in current frame (three passes, each on different criteria)
        labels = [0] * num_people
        
        
        # First, orientation
        properties = vel_orientation_array
        standard = params['orientation_threshold']
        
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
                db = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db.fit_predict(sub_properties)
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        
        
        # Second, velocity
        properties = vel_magnitude_array
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
                db = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db.fit_predict(sub_properties)
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        
        
        # Final, position
        properties = position_array
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
                db = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db.fit_predict(sub_properties)
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        
        
        return labels