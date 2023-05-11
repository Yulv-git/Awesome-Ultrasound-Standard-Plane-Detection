#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-20 18:17:37
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-03 16:41:05
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/AG-SonoNet/train_FPD_Classification.py
Description: FETAL_PLANES_DB Classification
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
from ipdb import set_trace

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from models import get_model


class StratifiedSampler(object):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.class_vector = class_vector
        self.batch_size = batch_size
        self.num_iter = len(class_vector) // 52
        self.n_class = 14
        self.sample_n = 2
        # create pool of each vectors
        indices = {}
        for i in range(self.n_class):
            indices[i] = np.where(self.class_vector == i)[0]

        self.indices = indices
        self.background_index = np.argmax([ len(indices[i]) for i in range(self.n_class)])


    def gen_sample_array(self):
        # sample 2 from each class
        sample_array = []
        for i in range(self.num_iter):
            arrs = []
            for i in range(self.n_class):
                n = self.sample_n
                if i == self.background_index:
                    n = self.sample_n * (self.n_class-1)
                arr = np.random.choice(self.indices[i], n)
                arrs.append(arr)

            sample_array.append(np.hstack(arrs))
        return np.hstack(sample_array)

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def train(args):
    # Load options
    json_opts = json_file_to_pyobj(args.config)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if args.debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    num_workers = train_opts.num_workers if hasattr(train_opts, 'num_workers') else 16
    train_dataset = ds_class(ds_path, split='train', transform=ds_transform['train'], preload_data=train_opts.preloadData, smoketest=args.smoketest)
    valid_dataset = ds_class(ds_path, split='val',   transform=ds_transform['valid'], preload_data=train_opts.preloadData, smoketest=args.smoketest)
    test_dataset  = ds_class(ds_path, split='test',  transform=ds_transform['valid'], preload_data=train_opts.preloadData, smoketest=args.smoketest)

    # create sampler
    if train_opts.sampler == 'stratified':
        print('stratified sampler')
        train_sampler = StratifiedSampler(train_dataset.labels, train_opts.batchSize)
        batch_size = 52
    elif train_opts.sampler == 'weighted2':
        print('weighted sampler with background weight={}x'.format(train_opts.bgd_weight_multiplier))
        # modify and increase background weight
        weight = train_dataset.weight
        bgd_weight = np.min(weight)
        weight[abs(weight - bgd_weight) < 1e-8] = bgd_weight * train_opts.bgd_weight_multiplier
        train_sampler = sampler.WeightedRandomSampler(weight, len(train_dataset.weight))
        batch_size = train_opts.batchSize
    else:
        print('weighted sampler')
        train_sampler = sampler.WeightedRandomSampler(train_dataset.weight, len(train_dataset.weight))
        batch_size = train_opts.batchSize

    # loader
    train_loader = DataLoader(dataset=train_dataset, num_workers=num_workers, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=num_workers, batch_size=train_opts.batchSize, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=num_workers, batch_size=train_opts.batchSize, shuffle=True)

    # Training Function
    track_labels = np.arange(len(train_dataset.label_names))
    model.set_labels(track_labels)
    model.set_scheduler(train_opts)
    
    if hasattr(model, 'update_state'):
        model.update_state(0)

    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))        
        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()

            if epoch == (train_opts.n_epochs-1):
                import time
                time.sleep(36000)

            if train_opts.max_it == epoch_iter:
                break

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            model.reset_results()
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):
                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                if train_opts.max_it == epoch_iter:
                    break

            # save validation error
            errors = model.get_accumulated_errors()
            if split == 'validation':
                valid_err = errors['CE']

            stats = model.get_classification_stats()
            print('epoch: {}\tACC: {}'.format(epoch, stats['accuracy']))

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        if hasattr(model, 'update_state'):
            model.update_state(epoch)

        # Update the model learning rate
        model.update_learning_rate(metric=valid_err, epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Classification Training Function')
    parser.add_argument('-c', '--config', default='./configs/config_sononet_8_FPD.json', help='training config file')
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('--smoketest', type=bool, default=False)
    args = parser.parse_args()

    train(args)
