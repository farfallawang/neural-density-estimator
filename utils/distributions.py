#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:14:42 2019

@author: mengdie
"""

from __future__ import print_function
import torch
import torch.utils.data
import math

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.-1e-5

PI = torch.FloatTensor([math.pi])
PI.requires_grad = False
if torch.cuda.is_available():
    PI = PI.cuda()

# N(x | mu, var) = 1/sqrt(2pi*var) * exp[-1/(2var) (x-mean)(x-mean)]
# log N(x| mu, var) = - 0.5 * log(2pi) - 0.5 * log(var) - 0.5 * (x-mean)(x-mean)/var

def log_normal_diag(x, mu, log_var, reduce = True, dim = None):
    log_norm = -0.5 * (log_var + (x - mu) * (x - mu) * log_var.exp().reciprocal())
    if (reduce == True):
        log_norm = torch.sum(log_norm, dim)
    return log_norm
    
def log_normal_normalized(x, mu, log_var, reduce = True, dim = None):
    exp = (x - mu) * (x - mu) / log_var.exp()
    log_norm = -0.5 * torch.log(2 * PI) - 0.5 * log_var - 0.5 * exp
    if (reduce == True):
        log_norm = torch.sum(log_norm, dim)
    return log_norm

def log_normal_standard(x, reduce = True, dim = None):
    log_norm = -0.5 * torch.log(2. * PI) - 0.5 * x * x
    if (reduce == True):
        log_norm = torch.sum(log_norm, dim)
    return log_norm
    
    

