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

class DBSCAN(object):

    #def __init__(self,x,epsilon,minpts):   # x: 입력, epsilon: 최소 이웃 반경  minpts: 최소 이우 ㅅ개수
    def __init__(self,x, y,epsilon,minpts):   # x: pose_list  y: poly_speed_list of humans
        # The number of input dataset
        self.n = len(x)
        #print('x:',x.shape)
        # Euclidean distance
        p, q = np.meshgrid(np.arange(self.n), np.arange(self.n))   # 격자 그리드 만들기
        self.dist = np.sqrt(np.sum(((x[p] - x[q])**2),2))
        # Relative velocity
        self.dist_vel = np.sqrt(np.sum(((y[p] - y[q])**2),2))
        hyper_params = [5]
        #self.dist = (np.add(self.dist,np.multiply(hyper_params,self.dist_vel)))/(np.add([1],hyper_params))   # option 1 (나누는 숫자가 너무 커서 전반적을 작아짐(관대해짐))
        self.dist = (np.add(self.dist,np.multiply(hyper_params,self.dist_vel)))/2         # option 2   (나누는 숫자 고정)

        

        
        
        
        #self.dist = (self.dist + hyper_params*self.dist_vel)/(1+hyper_params)
        
        #print('길이:',self.n)
        #print('p,q:',p,q)
        #print('거리:',self.dist)
        # label as visited points and noise
        self.visited = np.full((self.n), False)     # 방문 flag
        self.noise = np.full((self.n),False)        # 잡음 인덱스  -> 한번 연산된 객체와 잡음 객체는 다시 연산되지 않도록
        # DBSCAN Parameters
        self.epsilon = epsilon
        self.minpts = minpts
        # Cluster
        self.idx = np.full((self.n),0)   # 클러스터링 이전 모든 agent의 초기 index는 0으로 초기화
        self.C = 0
        self.input = x

    # 클러스터링 1단계
    def run(self):
        # Clustering
        for i in range(len(self.input)):          # 입력데이터 x를 스캔하여
            if self.visited[i] == False:          # 방문 flag를 체크. false일경우,
                self.visited[i] = True
                self.neighbors = self.regionQuery(i)       # regionQuery함수로 군집 밀도를 충족하는 지 판단
                if len(self.neighbors) >= self.minpts:     # 군집 요건을 만족하게 되면 군집 이웃 객체를 대상으로 RegionQuery를 다시 반복하기 위해
                    self.C += 1
                    self.expandCluster(i)                  # expnadCluster 함수를 실행해 줌
                else : self.noise[i] = True
        return self.idx,self.noise

    def regionQuery(self, i):
        g = self.dist[i,:] < self.epsilon
        Neighbors = np.where(g)[0].tolist()
        return Neighbors

    # 클러스터링 2단계
    def expandCluster(self, i):
        self.idx[i] = self.C
        k = 0
       
        while True:
            if len(self.neighbors) <= k:return
            j = self.neighbors[k]
            if self.visited[j] != True:
                self.visited[j] = True

                self.neighbors2 = self.regionQuery(j)
                v = [self.neighbors2[i] for i in np.where(self.idx[self.neighbors2]==0)[0]]

                if len(self.neighbors2) >=  self.minpts:
                    self.neighbors = self.neighbors+v

            if self.idx[j] == 0 : self.idx[j] = self.C
            k += 1

    def sort(self):
        
        cnum = np.max(self.idx)
        self.cluster = []
        self.noise = []
        for i in range(cnum):
           
            k = np.where(self.idx == (i+1))[0].tolist()
            self.cluster.append([self.input[k,:]])
       
        self.noise = self.input[np.where(self.idx == 0)[0].tolist(),:]
        return self.cluster, self.noise

    def plot(self):
        
        self.sort()
        fig,ax = plt.subplots()
        
        for idx,group in enumerate(self.cluster):
        
            ax.plot(group[0][:,0],
                    group[0][:,1],
                    marker='o',
                    linestyle='',
                    label='Cluster {}'.format(idx))

            if self.noise.any() != None:
                ax.plot(self.noise[:,0],
                    self.noise[:,1],
                    marker='x',
                    linestyle='',
                    label='noise')

        ax.legend(fontsize=10, loc='upper left')
        plt.title('Scatter Plot of Clustering result', fontsize=15)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.show()


