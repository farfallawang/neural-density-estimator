#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:59:57 2019

@author: mengdie
"""

import torch
import torch.nn as nn

class MADE(nn.Module):
    def __init__(self, in_dim, out_dim, last_layer, activation):
        super(MADE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = nn.Parameter(torch.randn([in_dim, out_dim]))
        self.bias = nn.Parameter(torch.randn([out_dim]))
        #self.first_layer = first_layer
        self.last_layer = last_layer
        self.activation = activation
        
    def construct_mask(self, prev_connections):
        #sample maximum num of connections
#        if self.first_layer == True:
#            self.prev_connections = torch.randperm(self.in_dim) + 1
        max_connections = torch.zeros(self.out_dim)  
        for k in range(self.out_dim):
            tmp = torch.FloatTensor(1).uniform_(min(prev_connections), self.out_dim)
            max_connections[k] = torch.tensor(tmp.ceil(), dtype = torch.int64)
                   
        mask = torch.zeros([self.in_dim, self.out_dim])
        if self.last_layer == 1:
            for col in range(self.out_dim):
                for row in range(self.in_dim):
                    if max_connections[col] > prev_connections[row]:
                        mask[row, col] = 1 
        else:
            #mask = [1 if max_connections[col] > self.prev_connectins[row] for (row,col) in zip(self.in_dim, self.out_dim) ]
            for col in range(self.out_dim):
                for row in range(self.in_dim):
                    if max_connections[col] >= prev_connections[row]:
                        mask[row, col] = 1

        return mask, max_connections
    
    def forward(self, x, prev_connections):
            
        mask, max_connections = self.construct_mask(prev_connections)  
        masked_weights = self.weights * mask #in_dim * out_dim #torch.float32
        h_x = torch.matmul(x, masked_weights) + self.bias
                
        return h_x, max_connections
    

class DeepMADE(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_units, activation, num_layers, prev_connections):  
        super(DeepMADE, self).__init__()
        self.layers = num_layers
        self.prev_connections = prev_connections
        #self.in_dims = [in_dim, hidden_units]
        #self.out_dims = [hidden_units, out_dim]
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.last_layer = False #torch.tensor(False).repeat(self.layers-1) + torch.tensor(1)
        self.model = nn.Sequential(*(
                MADE(in_dim, out_dim, last_layer, activation) for in_dim, out_dim in zip(self.in_dims, self.out_dims)), #_ in range(self.layers)),
        )
    
    def forward(self, train_loader, prev_connections):
        for mod in self.model:
            x, max_connections = mod(train_loader, prev_connections)
            train_loader = x
            prev_connections = max_connections
        
        return x
        
    
    

#        self.num_layers = num_layers
#        self.hidden_units = hidden_units # is an array (num_layers * hidden)        
#        self.model = nn.Sequential(*(
#                MADE(self) for _ in self.num_layers),
#                nn.Sigmoid()
#        )
        
#    def sample_maximum_connections(self):
#        m_0 = torch.randperm(self.input_size) + 1
#
#        for l in range(self.num_layers):
#            k_l = self.hidden_units[l]
#            m_l = torch.zeros(k_l)
#            for k in range(k_l):
#                m_l[k] = torch.FloatTensor().uniform_(min(m_0), self.input_size)
#            m_0 = m_l        
#        # combine arrays of m together        
#        return             