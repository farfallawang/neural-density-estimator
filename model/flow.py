#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:59:41 2019

@author: mengdie
"""
# Cite: https://github.com/ex4sperans/variational-inference-with-normalizing-flows/blob/master/flow.py

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.Tensor as tensor        
        
class PlanarFlow(nn.Module):
    def __init__(self, in_features):
        super(PlanarFlow, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, in_features))
        self.biases = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.randn(1, in_features)) 
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        activation = F.linear(z, self.weights, self.biases) # 512*1
        z = z + torch.mm(self.tanh(activation), self.scale)  # (512*1) * (1*2) 
        phi_z = torch.mm((1 - self.tanh(activation) ** 2), self.weights) # (512*1) (1*2) 
        log_det = 1 + torch.mm(phi_z, self.scale.t())  # (512 * 2) (2 * 1)
         
        return z, torch.log(torch.abs(log_det) + 1e-16)
   
class IAFFlow(nn.Module):
    def __init__(self, in_features):
        super(IAFFlow, self).__init__()
        
    def forward(self, z0, h_context):
        mu_, sigma_ = MaskedLinear(z0, h_context)
        z = z0 * sigma_ + mu_
        
        gate = F.sigmoid(sigma_)
        z = gate * z + (1-gate)*mu_
        log_det_jacobian = gate.log()
        
        return z, log_det_jacobian
        

#class RadialFlow(nn.Module):
#    def __init__(self, in_features):
#        super().__init__()
#        self.z0 = nn.Parameter(torch.randn(1, in_features))
#        self.alpha = nn.Parameter(torch.randn(1))
#        self.beta = - self.alpha + torch.log(1 + torch.exp(nn.Parameter(torch.randn(1))))
#        self.d = in_features
#        
#    def forward(self, z):
#        r = torch.norm(z - self.z0)  # 1
#        h = 1. / (self.alpha + r)  # 1
#        activation = self.beta * h * (z - self.z0) # 1 * dim
#        z = z + activation
#        #        r = torch.norm(z - self.z0)  # 1
#        h = 1. / (self.alpha + r)        
#        phi_h = -1./((self.alpha + r) **2) 
#        log_det = ((1 + self.beta * h) ** (self.d - 1)) * (1 + self.beta * h + self.beta * phi_h * r )
#
#        return z, torch.log(torch.abs(log_det) + 1e-16)

     