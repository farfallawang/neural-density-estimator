#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:17:37 2019

@author: mengdie
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import torch.autograd
import flow
from visualization import *
from target_density import *
import matplotlib.pyplot as plt

''' Build Flow ... '''

# Set constant
num_features = 2
batch_size = 512
epochs = 10000
lr = 1e-2
lr_decay = 0.999
flow_length = 6
true_density_choice = 2

# Generate sample from base distribution
base_dist = D.MultivariateNormal(torch.zeros(num_features), torch.eye(num_features))
z_0 = base_dist.sample([batch_size])

flow = NormalizingFlow(num_features, flow_length)
transformed_z, log_jacobs = flow(z_0)

# Visualize before training 
with torch.no_grad():
    visualize_flow(transformed_z)

# Set up loss    
z_k = transformed_z[-1]

if true_density_choice == 1:
    true_density = pot1
if true_density_choice == 2:
    true_density = pot2


def KL(base_dist, true_density, z_0, z_k, log_jacobs):
    log_q_k = base_dist.log_prob(z_0).unsqueeze(1) - sum(log_jacobs)
    log_p = torch.log(true_density(z_k) + 1e-16)
    loss = (log_q_k - log_p).mean() 
    return loss


''' Training ... '''

global_step = []
np_losses = []
optimizer = optim.Adam(flow.parameters(), lr = lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

for epoch in range(epochs): 
    scheduler.step()
    
    samples = base_dist.sample([batch_size])
    transformed_z, log_jacobs = flow(samples)
    z_k = transformed_z[-1]
    
    optimizer.zero_grad()
    loss = KL(base_dist, true_density, samples, z_k, log_jacobs)
    loss.backward(retain_graph = True)
    optimizer.step()
    
    if epoch % 1000 == 0:
        print("Loss on iteration " + "{:.0f}".format(epoch) + ": " + 
              "{:.4f}".format(loss.data))
        global_step.append(epoch)
        np_losses.append(loss.data)
    if epoch % 3000 == 0:
        with torch.no_grad():
            transformed_z, log_jacobs = flow(z_0)
            z_k = transformed_z[-1]
            plt.figure()
            plt.scatter(z_k[:,0], z_k[:,1], alpha = 0.7)
            plt.show()
        
start = 10
plt.plot(np_losses)        
    

''' Visualize after training ... '''
z_0 = base_dist.sample([batch_size])

with torch.no_grad():
    posterior_zk, _ = flow(z_0)

visualize_flow(posterior_zk)    
    
    
    
