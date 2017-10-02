from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from kitti_pcd_dataset import *
from point_feature import *
import torch.nn.functional as F
import time



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--cuda', type=bool, default = True,  help='cuda')

opt = parser.parse_args()
print (opt)

# cuda setting
if opt.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# print form
blue = lambda x:'\033[94m' + x + '\033[0m'

# random seed setting
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Dataset loading
print("start loading dataset...")
dataset = KittiPcdDataset(
    root='/data-sdb/chen01.lin/data/KITTI/pcd_extend', 
    train=True)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=opt.batchSize,
    shuffle=True, 
    num_workers=int(opt.workers), 
    collate_fn=collate_fn)
test_dataset = KittiPcdDataset(
    root='/data-sdb/chen01.lin/data/KITTI/pcd_extend', 
    train=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=opt.batchSize,
    shuffle=True, 
    num_workers=int(opt.workers), 
    collate_fn=collate_fn)
print("finish loading dataset!!!")

# output dir
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# point_feature module and init
print('Setting up models...')
feat_extr = PointNetfeat(k_nn = 5)
def xavier(param):
    init.xavier_uniform(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
if opt.model != '':
    feat_extr.load_state_dict(torch.load(opt.model))
else:
    feat_extr.apply(weights_init)
if opt.cuda:
    feat_extr = feat_extr.cuda()
print("model setings done")

# point_feature optimizer
optimizer = optim.SGD(feat_extr.parameters(), lr=1., momentum=0.9)

# train loop
num_batch = len(dataset)/opt.batchSize
for epoch in range(opt.nepoch):
    start = time.time()
    avg_batchproc_t = 0.
    
    for i, data in enumerate(dataloader, 0):
        
        # timing
        batchproc_t = time.time() - start
        avg_batchproc_t = (avg_batchproc_t * i + batchproc_t) / (i + 1.0)

        pointsets, indices = data
        pointfeats = []
        for i in xrange(opt.batchSize):
            pointset = pointsets[i]
            indice = indices[i]
            pointfeat = feat_extr(pointset, indice)
            pointfeats.append(pointfeat)
        pointfeats = torch.stack(pointfeats)

        target = Variable(torch.from_numpy(np.zeros([48, 160, 256], dtype=float)))
        optimizer.zero_grad()
        loss = F.l1_loss(pointfeats, pointfeats)
        params = feat_extr.parameters()
        
        # test
        # for param in params:
        #     print(param.grad)
        #     print(param.data)
        #     loss.backward()
        #     optimizer.step()
        #     print(param.grad)
        #     print(param.data)
        #     break

        
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(opt.batchSize)))

        if i % 10 == 0:
            print("batch processing time: %f" %avg_batchproc_t)
            j, data = enumerate(testdataloader, 0).next()
            points, target = data
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(2,1)
            points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.data[0], correct/float(opt.batchSize)))

        start = time.time()

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
