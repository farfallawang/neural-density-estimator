#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:26:44 2019

@author: mengdie
"""

import torch
from flow import *
import matplotlib.pyplot as plt
import numpy as np

def visualize_flow(samples):
    f, arr = plt.subplots(1, len(samples), figsize = (4*len(samples), 4))
    X0 = samples[0]
    for i in range(len(samples)):
        X1 = samples[i]
        idx = np.logical_and(X0[:,0] < 0, X0[:,1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s = 10, color = 'red')
        idx = np.logical_and(X0[:,0] < 0, X0[:,1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s = 10, color = 'green')
        idx = np.logical_and(X0[:,0] > 0, X0[:,1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s = 10, color = 'blue')
        idx = np.logical_and(X0[:,0] > 0, X0[:,1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s = 10, color = 'black')
        arr[i].set_xlim([-10, 10])
        arr[i].set_ylim([-10, 10])


#batch_size = 51
#flow_length = 6 
#'''Visualize before training... ''' 
#base_dist = D.MultivariateNormal(torch.zeros(2), torch.eye(2))
#x = base_dist.sample([batch_size])
#samples = [x]
#
#planar_flow = PlanarFlow(2)
#with torch.no_grad():
#    for i in range(flow_length):
#        x = planar_flow(x)
#        samples.append(x)
#
#visualize_flow(samples)



