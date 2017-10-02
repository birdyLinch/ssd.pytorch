from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
from sklearn.neighbors import KDTree
import time


class KittiPcdDataset(data.Dataset):
    '''
    Kitti point cloud dataset,
    root folder contains all the pcd files,
    extra train.txt/val.txt file contains the train val split,
    each file arranged as:'x y z r imgx imgy\n' x num_points is the processed pcd by Boss Wan
    currently unchanged from dataset.py
    '''
    def __init__(self, root='/data-sdb/chen01.lin/data/KITTI/pcd_extend', npoints = 2500, train = True):
        self.npoints = npoints
        self.root = root
        self.datapath = []
        self.centers, self.unit = self.get_supix_center(1280, 384, 160, 48)

        if train:
            fn_file = open(os.path.join(root, 'ImageSets', 'train.txt'))
        else:
            fn_file = open(os.path.join(root, 'ImageSets', 'val.txt'))

        for line in fn_file:
            line = line.strip('\n')
            # TODO: fix point cloud for 000000
            if line == '000000':
                continue
            self.datapath.append(os.path.join(root, line + '.txt'))

        # print(self.datapath)


    def __getitem__(self, index):
        '''
        produce a single data, label pair in numpy array format
        '''
        fn = self.datapath[index]
        '''
        This np.loadtxt() function is bloody hell useful
        '''
        point_set = np.loadtxt(fn).astype(np.float32)
        indice = self.supix_selection(point_set, k_nn=5)
        # point_set = torch.from_numpy(point_set)
        # indice = torch.from_numpy(indice.astype(np.int64))
        
        return point_set, indice

    def __len__(self):
        return len(self.datapath)

    def supix_selection(self, point_set, k_nn=5, scale=1.0):
        # use kd tree for nearest neighbor search
        kd_tree = KDTree(point_set[:, -2:], leaf_size=30, metric='euclidean')
        distance, indice = kd_tree.query(self.centers.reshape(-1, 2), k=k_nn, return_distance=True)
        # TODO: how to deal with no points in a close distance
        # indice[distance > scale * self.unit] = -1
        return indice.reshape(self.centers.shape[0], self.centers.shape[1], k_nn)

    def get_supix_center(self, max_x, max_y, supix_x=160, supix_y=48):
        # get supper pixel center (c_x, c_y) * N_pixel
        c_x = np.arange(0, supix_x)
        c_y = np.arange(0, supix_y)
        unit_x = max_x / supix_x
        unit_y = max_y / supix_y
        c_x = (c_x + 0.5) * unit_x
        c_y = (c_y + 0.5) * unit_y
        centers = np.dstack(np.meshgrid(c_x, c_y))
        print('shape of the centers: ', centers.shape)
        return centers, (unit_x + unit_y / 2.) 

def collate_fn(batch):
    data_batch = []
    indice_batch = []
    for sample in batch:
        data, indice = sample
        data_batch.append(data)
        indice_batch.append(indice)
    return data_batch, indice_batch



if __name__ == '__main__':
    print('test')
    d = KittiPcdDataset(root = '/data-sdb/chen01.lin/data/KITTI/pcd_extend')
    start = time.time()
    ps, seg = d[0]
    print('processing time: ', time.time()- start)
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = KittiPcdDataset(root = '/data-sdb/chen01.lin/data/KITTI/pcd_extend')
    start = time.time()
    ps, cls = d[1]
    print('processing time: ', time.time()- start)
    print(ps.size(), ps.type(), cls.size(),cls.type())
