"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
KITTIroot = '/data-sdb/chen01.lin/ssd/KITTIdevkit'
from kitti_pcd_dataset import KITTI_CLASSES as labelmap
import torch.utils.data as data
from utils.augmentations import SSDAugmentation

from kitti_pcd_dataset import AnnotationTransform, KittiPcdDataset, KITTI_CLASSES, BaseTransform
from pcd_ssd import build_ssd
from layers.modules import MultiBoxLoss
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
from point_feature import *

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model_ssd', default='weights/pcdssd_ssdnet_100000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--kitti_root', default='/data-sdb/chen01.lin/ssd/KITTIdevkit', help='Location of VOC root directory')
parser.add_argument('--trained_model_featextr', default='weights/pcdssd_featextr_100000.pth')
parser.add_argument('--trained_model_pcd_process', default='weights/pcdssd_pcdproclayer_100000.pth')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# annopath = os.path.join(args.kitti_root, 'KITTI', 'Annotations', '%s.xml')
# imgpath = os.path.join(args.kitti_root, 'KITTI', 'JPEGImages', '%s.png')
# imgsetpath = os.path.join(args.kitti_root, 'KITTI', 'ImageSets', 'Main', '{:s}.txt')

devkit_path = args.kitti_root
dataset_mean = (104, 117, 123)
set_type = 'train'

criterion = MultiBoxLoss(3, 0.5, True, 0, True, 3, 0.5, False, True)
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def test_net(save_folder, net, pcd_process, feat_extr, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    # """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        pcd, indice, im, gt, fns = dataset[3]
        im, gt, h, w = dataset.pull_item(3)
        ori_img = dataset.pull_image(3)

        x = Variable(im.unsqueeze(0))
        targets = [Variable(torch.from_numpy(gt).cuda(), volatile=True)]
        targets[0] = targets[0].float()
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        
        pointfeats = []
        pointfeat = feat_extr(pcd, indice, args.cuda)
        # print(pointfeat)
        # print(feat_extr.state_dict().keys())
        # print(feat_extr.state_dict()['conv1.weight'])
        # exit()
        # print(feat_extr)
        # print(pointfeat.data.shape)
        pointfeats.append(pointfeat)
        pointfeats = torch.stack(pointfeats)
        pointfeats = pointfeats.permute(0, 3, 1, 2)
        # print(pointfeats)
        # print(x)
        # cv2.imshow('img', x[0].permute(1, 2, 0).data.cpu().numpy().astype(np.uint8))
        # cv2.waitKey(0)
        # exit()
        out = net(x, pointfeats, pcd_process)
        # print(targets)
        # print(out[0])
        # exit()
        # detections = out.data
        loss_l, loss_c, N, diff = criterion(out, targets)
        print(N)
        print(loss_l, loss_c)
        print(gt.shape)
        exit()
        print(detections.shape)
        # img = (im.numpy()).astype(np.uint8)
        # img = np.stack([img[2], img[1], img[0]])
        # img = np.transpose(img, (1, 2, 0))
        # print(gt)
        # print(detections[[]])
        # cv2.imshow('img', ori_img)
        # cv2.waitKey(0)


        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            print(dets)
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets
        print(all_boxes[2][:10])
        gt_boxes = gt[:, :4]
        gt_boxes[:, 0] *= w
        gt_boxes[:, 2] *= w
        gt_boxes[:, 1] *= h
        gt_boxes[:, 3] *= h
        print(gt)
        cv2.imshow('img', ori_img)
        cv2.waitKey(0)


        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
        exit()
    exit()

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    all_boxes = pickle.load(open(det_file, 'rb'))

    print('Evaluating detections')


if __name__ == '__main__':
    # load net
    num_classes = len(KITTI_CLASSES) + 1 # +1 background
    print(num_classes)
    net = build_ssd('train', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model_ssd))
    feat_extr = PointNetfeat(k_nn=5)
    feat_extr.load_state_dict(torch.load(args.trained_model_featextr))
    pcd_process = torch.nn.Sequential(
                    torch.nn.Conv2d(256, 512, 3, stride=2, padding=1), 
                    # torch.nn.MaxPool2d(kernel_size=2, stride=2)
                )
    pcd_process.load_state_dict(torch.load(args.trained_model_pcd_process))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = KittiPcdDataset(args.kitti_root+'/KITTI', ['train'], SSDAugmentation(300, dataset_mean), AnnotationTransform())
    # KittiPcdDataset(args.kitti_root, train_sets, SSDAugmentation(
    #     ssd_dim, means), AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        feat_extr = feat_extr.cuda()
        pcd_process = pcd_process.cuda()
        # cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, pcd_process, feat_extr, args.cuda, dataset,
             BaseTransform(1280, 384, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
