#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:59:06 2019

@author: mengdie
"""

import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from model.flow import *

class VAE(nn.Module):
    def __init__(self, in_features, z_size, intermediate_dim):
        super(VAE, self).__init__()
        self.in_features = in_features
        self.z_size = z_size
        self.intermediate_dim = intermediate_dim
        
        self.q_z_nn = self.create_encoder() 
        self.p_x_nn = self.create_decoder() 
        
        self.q_z_nn_output_dim = 256
        
        self.log_det_j = torch.FloatTensor(1).zero_()
        self.log_det_j.requires_grad = True
    
    def create_encoder(self):
        """returns mean and variance of encoded z """
        q_z_nn = nn.Sequential(
                nn.Linear(self.in_features, self.intermediate_dim),
                nn.ReLU(),
                nn.Linear(self.intermediate_dim, self.z_size * 2),
                nn.Softplus()
                )
        
        return q_z_nn 
    
    def create_decoder(self):
        """returns mean and variance of decoded x """
        p_x_nn = nn.Sequential(
                nn.Linear(self.z_size, self.intermediate_dim),
                nn.ReLU(),
                nn.Linear(self.intermediate_dim, self.in_features),
                nn.Sigmoid()  # will explode if set to ReLU()
                )
            
        return p_x_nn  
    
    def reparametrize(self, mu, var):
        std = var.sqrt()
        eps = torch.FloatTensor(var.size()).normal_()
        eps.requires_grad = True
        z = eps.mul(std).add_(mu)
        
        return z
    
    def encode(self, x):
        """ x: batch_size * num_channels * width * height """
        
        h = self.q_z_nn(x)  #(batch_size, z_size * 2)
        h = h.view(h.size(0), -1) 
        mean = h[:, :self.z_size]  
        var = h[:, self.z_size:]
        return mean, var
                
    def decode(self, z):
        """ """
        z = z.view(z.size(0), self.z_size)
        x_mean = self.p_x_nn(z)
        return x_mean 
        
    def forward(self, x):
        """ """
        
        z_mu, z_var = self.encode(x)
        # Sample z_0
        z = self.reparametrize(z_mu, z_var)
        x_mean = self.decode(z)
        
        return x_mean, z_mu, z_var, self.log_det_j, z, z 
        
class PlanarVAE(VAE):
    """ VAE with planar flows in the encoder """
    def __init__(self, in_features, z_size, intermediate_dim, num_flows): #, num_flows, batch_size
        super(PlanarVAE, self).__init__(in_features, z_size, intermediate_dim) #, num_flows, batch_size
        self.log_det_j = 0.
        self.in_features = in_features
        self.z_size = z_size
        self.intermediate_dim = intermediate_dim
        self.num_flows = num_flows
        self.transforms = nn.Sequential(*(
                PlanarFlow(self.z_size) for _ in range(self.num_flows)
                ))
                
    def forward(self, x):
        z_mu, z_var = self.encode(x)
        # Sample z_0
        z = self.reparametrize(z_mu, z_var)
        z_0 = z
        
        for transform in self.transforms:
            z, log_det_jacobian = transform(z)
            self.log_det_j += log_det_jacobian
        
        z_k = z
        x_mean = self.decode(z_k)
        
        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k
        
        
class IAFVAE(VAE):
    def __init__(self, in_features, z_size, intermediate_dim, num_flows):
        super(IAFVAE, self).__init__(in_features, z_size, intermediate_dim)
        self.log_det_j = 0
        self.in_features = in_features
        self.z_size = z_size
        self.intermediate_dim = intermediate_dim
        self.transforms = nn.Sequential(*(
                IAFFlow(self.z_size) for _ in range(self.num_flows)
                ))
        
    def encode(self, x):
        h = self.q_z_nn(x)  #(batch_size, z_size * 2)
        h = h.view(h.size(0), -1) 
        mean = h[:, :self.z_size]  
        var = h[:, self.z_size:]
        return mean, var, h
            
    def forward(self, x):
        z_mu, z_var, h_context = self.encode(x)
        # Sample z_0
        z = self.reparametrize(z_mu, z_var)
        z_0 = z
        
        for transform in self.transforms:
            z, log_det_jacobian = transform(z, h_context)
            self.log_det_j += log_det_jacobian
        
        z_k = z
        x_mean = self.decode(z_k)
        
        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k
        
        