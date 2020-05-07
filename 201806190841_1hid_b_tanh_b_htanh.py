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

Predict audio envelope for output samples using all 64 ch. eeg

Reference:
https://github.com/tdetaillez/neural_networks_auditory_attention_decoding/blob/master/Code.py
    
"""
__author__ = 'Greg Ciccarelli'
__date__ = 'June 19, 2018'

import numpy as np
import torch
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F    

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()

        # i2o creates the output sample
        self.i2o_l1 = nn.Linear(input_size, hidden_size, bias=True)
        self.i2o_l2 = nn.Linear(hidden_size, output_size, bias=True)
        
        self.bn_l0 = nn.BatchNorm1d(input_size)
        self.bn_l1 = nn.BatchNorm1d(hidden_size)        
     

    def forward(self, x):
        #print(x.size())

        x = x.view(x.shape[0], -1)
        #print('-after reshape-')
        #print(x.size())
        
        x = self.bn_l0(x)      
        
        x = self.i2o_l1(x)
        x = F.tanh(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = self.bn_l1(x)
        
        x = F.hardtanh(self.i2o_l2(x))
        output = x

        return output
