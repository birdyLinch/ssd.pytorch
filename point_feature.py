from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import pdb
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, k_nn=5):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 256, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.mp1 = torch.nn.MaxPool1d(k_nn)
        self.k_nn = k_nn
    def forward(self, pointset, indice, cuda):
        # NOTE: this forward function of class PointNetfeat takes (list of) numpy array as input
        # x is (B, n_points, 4)
        # indice is (B, supix_y, supix_x, k_nn)
        supix_y, supix_x, _ = indice.shape
        x = pointset[indice]

        if cuda:
            x = Variable(torch.from_numpy(x).cuda().view(-1, self.k_nn, 6).transpose(2, 1))
        else:
            x = Variable(torch.from_numpy(x).view(-1, self.k_nn, 6).transpose(2, 1))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
       	pointfeat = self.mp1(x)
       	pointfeat = pointfeat.view(supix_y, supix_x, 256)
       	# print('point feature shape is: ', pointfeat.data.shape)
       	return pointfeat
