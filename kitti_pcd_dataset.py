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
# for image dataset
import cv2
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def base_transform(image, size_x, size_y, mean):
    x = cv2.resize(image, (size_x, size_y)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size_x, size_y, mean):
        self.size_x = size_x
        self.size_y = size_y
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size_x, self.size_y, self.mean), boxes, labels

# KITTI_CLASSES = ('dontcare' ,'pedestrian', 'person_sitting', 'cyclist', 'car', 'van', 'truck','tram', 'misc')

# Note Van is not counted as a negetive classes and dontcare objects will be removed in loss
KITTI_CLASSES = ('dontcare', 'car')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = dict(zip(KITTI_CLASSES, range(len(KITTI_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            # TODO: isolate bad hard example
            # difficult = int(obj.find('difficult').text) == 1
            # if not self.keep_difficult and difficult:
            #     continue
            difficult = 0
            name = obj.find('name').text.lower().strip()
            
            # filter the negetive classes
            if name not in KITTI_CLASSES:
                continue

            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax','x','y','z','h','w','l','alpha','ry']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) # - 1 ? TODO
                # scale height or width
                if i == 0 or i==2:
                    cur_pt = cur_pt / width  
                if i == 1 or i==3:
                    cur_pt = cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class KittiPcdDataset(data.Dataset):
    '''
    Kitti point cloud dataset,
    root folder contains all the pcd files,
    extra train.txt/val.txt file contains the train val split,
    each file arranged as:'x y z r imgx imgy\n' x num_points is the processed pcd by Boss Wan
    currently unchanged from dataset.py
    '''
    def __init__(self, root, image_sets, transform=None, target_transform=None, dataset_name='KITTI'):
        self.root = root
        self._pcdpath = os.path.join(root, 'PointClouds', '%s.txt')
        self._imgpath = os.path.join(root, 'Images', '%s.png')
        self._annopath = os.path.join(root, 'Annotations', '%s.xml')
        self.ids = list()
        self.transform = transform
        self.target_transform = target_transform
        self.name = 'KITTI'
        self.centers, self.unit = self.get_supix_center(1280, 384, 160, 48)

        for name in image_sets:
            fn_file = open(os.path.join(root, 'ImageSets', name + '.txt'))
            for line in fn_file:
                line = line.strip()
                # TODO: fix point cloud for 000000
                if line == '000000':
                    continue
                self.ids.append(line)

        # print('see if the dataset id found properly' self.ids)


    def __getitem__(self, index):

        '''
        produce a single data, label pair in numpy array format
        '''
        point_set, indice, fn = self.get_pcd_and_indice(index)
        im, gt, h, w = self.pull_item(index)
        
        return point_set, indice, im, gt, fn

    def __len__(self):
        return len(self.ids)

    ###################################
    # point cloud processing functions
    #
    ###################################
    def get_pcd_and_indice(self, index):
        '''
        produce a single pcd, indice pair in numpy array format
        point_set : [num_points x 6] second dimention is consist of (x, y, z, ref, imgx, imgy)
        indice : [48, 168, k_nn]
        '''
        fn = self._pcdpath % self.ids[index]

        '''
        This np.loadtxt() function is bloody hell useful
        '''
        point_set = np.loadtxt(fn).astype(np.float32)
        indice = self.supix_selection(point_set, k_nn=5)
        
        return point_set, indice, fn

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

    #########################################
    # image, ground truth processing function
    #
    #########################################
    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            # if len(target) > 0:
            target = np.array(target)
            if target.shape[0] == 0: 
                target = np.array([[0.,]*13])
            # print(target, target.shape)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, :13])
            # cv2.imshow('img', img.astype(np.uint8))
            # cv2.waitKey(0)
            # to rgb           
            img = img[:, :, (2, 1, 0)]

            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, labels))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

def collate_fn(batch):
    targets = []
    imgs = []
    data_batch = []
    indice_batch = []
    fn_batch = []
    for sample in batch:
        data, indice, im, gt, fn = sample
        data_batch.append(data)
        indice_batch.append(indice)
        imgs.append(im)
        targets.append(torch.FloatTensor(gt))
        fn_batch.append(fn)
    return data_batch, indice_batch, torch.stack(imgs, 0), targets, fn_batch 

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    data_batch = []
    indice_batch = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets



if __name__ == '__main__':
    print('test')
    d = KittiPcdDataset(root='/data-sdb/chen01.lin/ssd/KITTIdevkit/KITTI', image_sets=['train'])
    start = time.time()
    pcd, indice, img, gt = d[0]
    print('processing time: ', time.time()- start)
    print(pcd, img, type(pcd), type(img))

    d = KittiPcdDataset(root = '/data-sdb/chen01.lin/data/KITTI/pcd_extend')
    start = time.time()
    ps, cls = d[1]
    print('processing time: ', time.time()- start)
    print(ps.size(), ps.type(), cls.size(),cls.type())
