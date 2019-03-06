#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:04:38 2019

@author: mengdie
"""

import torch


def safe_log(tensor):
    return torch.log(tensor + 1e-16)