#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:25:13 2019

@author: mengdie
"""

import torch
import torch.nn as nn
#from model.MADE import *
from model.MADE import *
import torch.optim as optim
from utils.load_data import *
from utils.utils import *
from training.training import *
import numpy as np        
import pandas as pd

# Set Constant
input_size = 784
num_layers = 2
hidden_units = [300, 100]
out_dim = 784
learning_rate = 1e-3
batch_size = 100
epoch = 2
prev_connections = torch.randperm(input_size).float() + 1

in_dims = [784, 300, 100]
out_dims = [300, 100, 10]


# Set up data
data = torch.tensor(np.asarray(pd.read_csv("tmp.csv", skiprows = 0))).float()

# Train on MADE
#model = MADE(input_size, hidden_units[0], False, "ReLU")
#mask, max_con = model.construct_mask(prev_connections)
#x, max_con = model.forward(data, prev_connections)

# Train on DeepMADE
model = DeepMADE(in_dims, out_dims, hidden_units, "Relu", num_layers, prev_connections)
print(model)
x = model.forward(data, prev_connections) 

## Construct MADE model
#model = nn.Sequential(
#        MADE(input_size, hidden_units[0], False, "ReLU"),  
#        MADE(hidden_units[0], hidden_units[1], False, "ReLU"),
#        MADE(hidden_units[1], 10, True, "ReLU")
#        #nn.Sigmoid()
#        )
#
#print(model)
#
#
#
#for mod in model:
#    x, max_connections = mod(train_loader, prev_connections)
#    train_loader = x
#    prev_connections = max_connections
#    
#
#train_loss = np.zeros(len(train_loader)) # batch_size
##beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
##print('beta = {:5.4f}'.format(beta))
#
#data_lst = []
#for batch_idx, (data, _) in enumerate(train_loader):
#    data.requires_grad = True   #(batch_size, 784)
#    x, max_connections = model(data, prev_connections)
#    z_log_var = safe_log(z_var)
##        if model_name == 'VAE':
##            z_log_var = safe_log(z_var)
##            loss = vlb_binomial(data, x_mean, z_mu, z_log_var)
##        if model_name == 'PlanarVAE':
#    loss = binary_loss_function(data, x_mean, z_mu, z_log_var, log_det_j, z_0, z_k)
#    train_loss[batch_idx] = loss
#    loss.backward(retain_graph = True)
#    
#    opt.step()
#


# Construct MADE model version 2
#model = MADE()


#opt = optim.Adamax(model.parameters(), lr = learning_rate, eps = 1.e-7)
#train_loss = train(train_loader, model, opt, input_size, epoch)

#
#
#
#optimizer.zero_grad()
#
## Set up loss
#
#p_x = data * safe_log(x_decoded) + (1-data) * safe_log(1-x_decoded).sum(dim = 1).exp()
#loss = -safe_log(p_x)
#
#loss.backward(retain_graph = True)
#optimizer.step()

# Train