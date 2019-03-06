#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:17:36 2019

@author: mengdie
"""

import torch
import torch.nn as nn

class MADE(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_units):
        super(MADE, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        #self.weights = nn.Parameter(torch.randn([in_dim, out_dim]))
        #self.bias = nn.Parameter(torch.randn([out_dim]))
        #self.first_layer = first_layer
        self.hidden_units = hidden_units
        #self.last_layer = last_layer
        #self.activation = activation
        #self.prev_connections = prev_connections

    def construct_mask(self, prev_connections):
        #sample maximum num of connections
#        if self.first_layer == True:
#            self.prev_connections = torch.randperm(self.in_dim) + 1
        max_connections = torch.zeros(self.out_dim)  
        for k in range(self.out_dim):
            max_connections[k] = torch.FloatTensor().uniform_(min(prev_connections), self.out_dim)
                   
        mask = torch.zeros([self.in_dim, self.out_dim])
        if self.last_layer == True:
            for col in range(self.out_dim):
                for row in range(self.in_dim):
                    if max_connections[col] >= prev_connections[row]:
                        mask[row, col] = 1 
        else:
            #mask = [1 if max_connections[col] > self.prev_connectins[row] for (row,col) in zip(self.in_dim, self.out_dim) ]
            for col in range(self.out_dim):
                for row in range(self.in_dim):
                    if max_connections[col] >= prev_connections[row]:
                        mask[row, col] = 1

        return mask, max_connections
    

    
    def forward(self, x, prev_connections):
        in_dim = self.in_dims
        for l in range(self.num_layers):
            out_dim = self.hidden_units[l]
            weights = nn.Parameter(torch.randn([in_dim, out_dim]))
            bias = nn.Parameter(torch.randn([out_dim]))
            mask, max_connections = self.construct_mask(prev_connections)              
            masked_weights = weights * mask #in_dim * out_dim
            h_x = torch.mat_mul(x, masked_weights) + bias
            prev_connections = max_connections
            x = h_x
            in_dim = self.hidden_units[l]
        
        
        
        return h_x