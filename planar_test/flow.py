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

class NormalizingFlow(nn.Module):
    def __init__(self, in_features, flow_length):
        super().__init__()
        self.transforms = nn.Sequential(*(
                PlanarFlow(in_features) for _ in range(flow_length)
        ))
    
        self.log_det_jacobians = nn.Sequential(*(
                PlanerFlowLogDetJacobian(t) for t in self.transforms
        )) 

            
    def forward(self, z):
        log_jacob = []
        transformed_z = [z]
        for transform, log_det_jacobian in zip(self.transforms, self.log_det_jacobians):
            log_jacob.append(log_det_jacobian(z))
            z = transform(z)
            transformed_z.append(z)
        #z_k = z
        
        return transformed_z, log_jacob
        
        
class PlanarFlow(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, in_features))
        self.biases = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.randn(1, in_features)) 
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        activation = F.linear(z, self.weights, self.biases) # 512*1
        z = z + torch.mm(self.tanh(activation), self.scale)  # (512*1) * (1*2) 
        return z  
   

class PlanerFlowLogDetJacobian(nn.Module):
    def __init__(self, affine):
        super().__init__()
        self.weights = affine.weights 
        self.biases = affine.biases
        self.scale = affine.scale
        self.tanh = affine.tanh
       
    def forward(self, z):
        activation = F.linear(z, self.weights, self.biases)  # 512 * 1
        phi_z = torch.mm((1 - self.tanh(activation) ** 2), self.weights) # (512*1) (1*2) 
        log_det = 1 + torch.mm(phi_z, self.scale.t()) #(512*2) * (2*1) 
        return torch.log(torch.abs(log_det) + 1e-16)        



     
