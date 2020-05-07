"""
This branch is the software release for the 2019 paper: https://www.nature.com/articles/s41598-019-47795-0

See LICENSE.txt

Copyright 2019 Massachusetts Institute of Technology

Useage:

import sys
from importlib import reload

# path to folder containing the class.py module
sys.path.append(os.path.split(file_path_name_net)[0])

module = __import__(os.path.split(file_path_name_net)[1]) 
reload(module) # handle case of making changes to the module- forces reload
NN = getattr(module, 'NN')

Construct a binary classification network that operates on one eeg channels and a bank of audio filters.
    
"""
__author__ = 'Greg Ciccarelli'
__date__ = 'April 4, 2018'

import numpy as np
import torch
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F    

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()

        # hidden_size is the window size
        self.hidden_size = hidden_size

        # weight matrix is (n_input, n_output) and a bias (n_output)
        
        kernel_size = 3
        stride = 1
        padding = 0
        dilation = 1
        in_channels = 65
        out_channels = 64        
        self.conv_3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True)
        
        self.m1 = nn.MaxPool1d(kernel_size, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        kernel_size = 1
        stride = 1
        padding = 0
        dilation = 1
        
        in_channels = 64
        out_channels = 2
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True)

        # i2o creates the output sample
        input_size = 100
        self.i2o_l1 = nn.Linear(2 * 123, 2 * (input_size), bias=True)
        self.i2o_l2 = nn.Linear(2 * (input_size), 2 * (input_size), bias=True)
        self.i2o_l3 = nn.Linear(2 * (input_size), 1 * (input_size), bias=True)        
        self.i2o_l4 = nn.Linear(1 * (input_size), 1 * (output_size), bias=True)  
        
        self.bn_l0 = nn.BatchNorm1d(65)
        self.bn_l1 = nn.BatchNorm1d(2 * 123)        
        self.bn_l2 = nn.BatchNorm1d(2 * (input_size))
        self.bn_l3 = nn.BatchNorm1d(2 * (input_size))         

    def forward(self, x):
        #print(x.size())
        #print(x)
        
        x = self.bn_l0(x)
        
        x = F.elu(self.conv_3(x))
        #print('-after conv3')
        #print(x.size())

        x = self.m1(x)
        #print('-after m1-')
        #print(x.size())

        x = F.elu(self.conv_1(x))
        #print('-after conv 1-')
        #print(x.size())
        x = x.view(x.shape[0], -1)
        #print('-after reshape-')
        #print(x.size())
        x = self.bn_l1(x)        
        x = self.i2o_l1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.bn_l2(x)          
        x = self.i2o_l2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.bn_l3(x)          
        x = self.i2o_l3(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.25, training=self.training)        

        x = self.i2o_l4(x)
        output = x

        return output
