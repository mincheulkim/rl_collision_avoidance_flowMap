import numpy as np

# -*- coding: utf-8 -*-
"""
Project Code: DBSCAN v1.1
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-09
Contact Info: :
    https://deep-eye.tistory.com
    https://deep-i.net
"""

from dbscan import DBSCAN
from scipy import io

#%% Run DEMO
x = io.loadmat('./sample/sample.mat')['X']   # sample data load
y = [[-6,-6],[-6,-7],[-7,-6],[-7,-7],[-7,-8],[-6,6],[-6,7],[-7,6],[8,0],[7,0]]
y = np.array(y)

# Initialize DBSCAN
#dbscan = DBSCAN(x,1.5,4)   # init papameters
dbscan = DBSCAN(y,1.5,3)   # init papameters
# Run DBSCAN(CLUSTERING)
idx,noise = dbscan.run()    # run DBSCAN
# Result SORTING
g_cluster,n_cluster = dbscan.sort()
# Visualization results
print('g_cluster:',g_cluster,'noise_cluster:',n_cluster)
dbscan.plot()