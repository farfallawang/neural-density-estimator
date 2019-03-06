#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:16:19 2019

@author: mengdie
"""

from __future__ import print_function
import torch

from torch.autograd import Variable
from utils.utils import *
import numpy as np
from training.loss import *

model_name = 'PlanarVAE'

def train(train_loader, model, model_name, opt, input_size, epoch):
    train_loss = np.zeros(len(train_loader)) # batch_size
    #beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
    #print('beta = {:5.4f}'.format(beta))
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data.requires_grad = True   #(batch_size, 784)
        opt.zero_grad()
        x_mean, z_mu, z_var, log_det_j, z_0, z_k = model(data)
        z_log_var = safe_log(z_var)
#        if model_name == 'VAE':
#            z_log_var = safe_log(z_var)
#            loss = vlb_binomial(data, x_mean, z_mu, z_log_var)
#        if model_name == 'PlanarVAE':
        loss = binary_loss_function(data, x_mean, z_mu, z_log_var, log_det_j, z_0, z_k)
        train_loss[batch_idx] = loss
        loss.backward(retain_graph = True)
        
        opt.step()

    tr_loss = train_loss.sum() / len(train_loader)
    print('====> Epoch: {:3d} Average train loss: {:.4f}'.format(
            epoch, tr_loss))    
    
    return tr_loss


def evaluate(data_loader, model, input_size):
    loss = 0.
    batch_idx = 0
    
    for data, _ in data_loader:
        batch_idx += 1
        
        data.requires_grad = False
        data = data.view(-1, input_size)
        
        x_mean, z_mu, z_var, log_det_j, z_0, z_k = model(data)
        z_log_var = safe_log(z_var)
        #batch_loss = vlb_binomial(data, x_mean, z_mu, z_log_var)
        batch_loss = binary_loss_function(data, x_mean, z_mu, z_log_var, log_det_j, z_0, z_k)
        loss += batch_loss
        
    loss /= len(data_loader)
    
    print('====> Validation set loss: {:.4f}'.format(loss))
    
    return loss