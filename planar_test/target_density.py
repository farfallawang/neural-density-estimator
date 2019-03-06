#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:46:13 2019

@author: wang5699
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as D


def pot1(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)

    exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)
    return torch.exp(-u)

def w1(z1):
    return torch.sin(2.*np.pi*z1/4.)

def w2(z1):
    return 3*torch.exp(-0.5*((z1-1.)/.6)**2)

def pot2(z):
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    return 0.5*((z2 - w1(z1))/0.4)**2
    


    



    
    
    


