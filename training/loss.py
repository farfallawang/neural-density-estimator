#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:28:24 2019

@author: mengdie
"""

import math
import torch 
from utils.distributions import *
from utils.utils import *



def binary_loss_function(x, x_decoded_mean, z_mu, z_log_var, log_det_j, z_0, z_k, beta = 1.):
    
    log_q_z0 = log_normal_diag(z_0, z_mu, z_log_var, dim = 1) 
    log_q_zk = log_q_z0 - log_det_j  #log_det_j would be zero for plain VAE
    log_p_zk = log_normal_standard(z_k, dim = 1) 
    
    # reconstruction loss
    log_p_x = - (x * safe_log(x_decoded_mean) + (1-x) * safe_log(1 - x_decoded_mean)).sum(1) 
    kl = log_q_zk - log_p_zk
    
    return (log_p_x + beta * kl).mean()


#def vlb_binomial(x, x_decoded_mean, t_mean, t_log_var):
#    """Returns the value of Variational Lower Bound
#    
#    The inputs are tf.Tensor
#        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
#        x_decoded_mean: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
#        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
#        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)
#    
#    Returns:
#        A tf.Tensor with one element (averaged across the batch), VLB
#    """
#    KL = 0.5* (-t_log_var + torch.exp(t_log_var)**2 + (t_mean ** 2) - 1).sum(1)
#    expectation = - (x * safe_log(x_decoded_mean) + (1-x) * safe_log(1 - x_decoded_mean)).sum(1) 
#     
#    return (expectation + KL).mean()

