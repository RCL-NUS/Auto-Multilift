"""
This file defines the class of neural network that parameterizes the hyperparameters
------------------------------------------------------------------------------------
Wang, Bingheng, 1st version: 12, Mar., 2024 at Control and Simulation Lab, ECE Dept. NUS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np

"""
[3] Bartlett, P.L., Foster, D.J. and Telgarsky, M.J., 2017.
    Spectrally-normalized margin bounds for neural network.
    Advances in neural information processing systems, 30.
"""

class Net(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        super(Net, self).__init__()
        self.linear1 = spectral_norm(nn.Linear(D_in, D_h)) # spectral normalization can stabilize DNN training and improve generalization to unseen data [3]
        self.linear2 = spectral_norm(nn.Linear(D_h,  D_h))
        self.linear3 = spectral_norm(nn.Linear(D_h,D_out))
        

    def forward(self, input):
        # convert the state input to a tensor
        S  = torch.tensor(input, dtype=torch.float) # column 2D tensor
        z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
        z2 = F.relu(z1) # hidden layer 1
        z3 = self.linear2(z2)
        z4 = F.relu(z3) # hidden layer 2
        z5 = self.linear3(z4) 
        z6 = torch.sigmoid(z5) # output layer (a row tensor)
        return z6.t()
    
    def myloss(self, para, dp):
        # convert a np.array to a tensor, para is the network's output which is a column tensor (z6.t())
        Dp = torch.tensor(dp, dtype=torch.float) # row 2D tensor, dp = dlids@Xi_pi or dp = dllds@Xl_pl, a row vector
        loss_nn = torch.matmul(Dp, para) # a scalar
        return loss_nn
    
    