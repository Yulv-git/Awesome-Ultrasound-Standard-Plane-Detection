#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-20 18:17:37
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-03 12:57:38
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/AG-SonoNet/dataio/loader/us_dataset.py
Description: Modify here please
Init from https://github.com/ozan-oktay/Attention-Gated-Networks
'''
from itertools import count
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, sampler
import h5py
import numpy as np
import datetime
from os import listdir
from os.path import join
import csv
from PIL import Image
from ipdb import set_trace

# from .utils import check_exceptions


class UltraSoundDataset(data.Dataset):
    def __init__(self, root_path, split='train', transform=None, preload_data=False, smoketest=False):
        super(UltraSoundDataset, self).__init__()

        f = h5py.File(root_path)
        self.images = f['x_'+split]

        if preload_data:
            self.images = np.array(self.images[:])

        self.labels = np.array(f['p_'+split][:], dtype=np.int64)  # [:1000]
        self.label_names = [x.decode('utf-8') for x in f['label_names'][:].tolist()]
        # print(self.label_names)
        # print(np.unique(self.labels[:]))
        # construct weight for entry
        self.n_class = len(self.label_names)
        class_weight = np.zeros(self.n_class)
        for lab in range(self.n_class):
            class_weight[lab] = np.sum(self.labels[:] == lab)
        class_weight = 1 / class_weight
    
        self.weight = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            self.weight[i] = class_weight[self.labels[i]]

        # print(class_weight)
        assert len(self.images) == len(self.labels)

        # data augmentation
        self.transform = transform

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        input  = self.images[index][0]
        target = self.labels[index]

        # input = input.transpose((1,2,0))

        # handle exceptions
        #check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)

        # print(input.shape, torch.from_numpy(np.array([target])))
        # print("target",np.int64(target))
        return input, int(target)

    def __len__(self):
        return len(self.images)


class UltraSoundDataset_FPD(data.Dataset):
    def __init__(self, root_path, split='train', transform=None, preload_data=False, smoketest=False):
        super(UltraSoundDataset_FPD, self).__init__()

        imgs_dir = '{}/Images'.format(root_path)
        labels_file = '{}/FETAL_PLANES_DB_data.csv'.format(root_path)
        labels_name2id = {"Other":0, "Maternal cervix":1, "Fetal abdomen":2, "Fetal femur":3, "Fetal thorax":4, "Trans-cerebellum":5, "Trans-thalamic":6, "Trans-ventricular":7}
        labels_items = csv.reader(open(labels_file, 'r'))
        labels = []
        self.images = []
        self.imgs_name = []
        count_tmp = 0
        for line_idx, labels_item in enumerate(labels_items):
            if line_idx == 0:
                continue
            labels_item = labels_item[0][:-1].split(";")
            if split=='train':
                if int(labels_item[-1]) != 1:
                    continue
            else:
                if int(labels_item[-1]) == 1:
                    continue
            count_tmp += 1
            if smoketest and count_tmp > 1024:
                break

            img_name = '{}/{}.png'.format(imgs_dir, labels_item[0])
            img = np.array(Image.open(img_name))
            # print('{}\t{}'.format(img.shape, img_name))
            if len(img.shape) > 2:  # To process dimensionally inconsistent data in an FPD dataset.
                img = img[:,:,0]
            if img.shape[0] < 208 or img.shape[1] < 272:
                continue  # AG-SonoNet's data reading process can lead to errors in the image of the wrong size.
            self.imgs_name.append(img_name)
            self.images.append(img)

            if labels_item[2] == "Fetal brain":
                labels.append(labels_name2id[labels_item[3]])
            else:
                labels.append(labels_name2id[labels_item[2]])

        if preload_data:
            self.images = np.array(self.images[:])

        self.label_names = list(labels_name2id.keys())
        self.labels = np.array(labels, dtype=np.int64)
        # construct weight for entry
        self.n_class = len(self.label_names)
        class_weight = np.zeros(self.n_class)
        for lab in range(self.n_class):
            class_weight[lab] = np.sum(self.labels[:] == lab)
        class_weight = 1 / class_weight
    
        self.weight = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            self.weight[i] = class_weight[self.labels[i]]

        assert len(self.images) == len(self.labels)

        # data augmentation
        self.transform = transform

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))  # train: 7129  val+test: 5271

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        input  = self.images[index]
        target = self.labels[index]

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)

        # print(input.shape, torch.from_numpy(np.array([target])))  # torch.Size([1, 208, 272]) tensor([2])
        
        return input, int(target)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = UltraSoundDataset("/vol/bitbucket/js3611/data_ultrasound/preproc_combined_inp_224x288.hdf5", 'test')

    ds = DataLoader(dataset=dataset, num_workers=1, batch_size=2)
