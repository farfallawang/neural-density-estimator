#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:38:13 2019

@author: wang5699
"""

from __future__ import print_function
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import torch.distributions as D
import os
import datetime
from model.VAE import *
from training.training import train, evaluate
from utils.load_data import *
from utils.utils import *


# SET CONSTANT

batch_size = 100
learning_rate = 1e-4
input_size = 784
z_size = 64
intermediate_dim = 300
epochs = 2000
early_stopping = 100
num_flows = 6
flow_name = "PlanarVAE"

''' LOAD DATA '''

train_loader, val_loader, test_loader = load_static_mnist(batch_size)

''' SELECT MODEL '''

#model = VAE(input_size, z_size, intermediate_dim)
model = PlanarVAE(input_size, z_size, intermediate_dim, num_flows)
print(model)

optimizer = optim.Adamax(model.parameters(), lr = learning_rate, eps = 1.e-7)

''' TRAINING '''

# for early stopping
best_loss = np.inf
best_bpd = np.inf

train_loss = []
val_loss = []

epoch = 0
train_times = []


for epoch in range(1, epochs + 1):
    t_start = time.time()
    tr_loss = train(train_loader, model, flow_name, optimizer, input_size, epoch)
    train_loss.append(tr_loss)
    
    v_loss = evaluate(val_loader, model, input_size)
    val_loss.append(v_loss)
    
    # early-stopping
    if v_loss < best_loss:
        e = 0
        best_loss = v_loss
        print('Model saved....')
        torch.save(model, 'test.model')
    
    
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name)   
