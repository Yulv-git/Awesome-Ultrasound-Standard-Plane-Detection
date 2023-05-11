#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-20 18:17:37
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-01 17:49:53
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/AG-SonoNet/train_classifaction.py
Description: Modify here please
Init from https://github.com/ozan-oktay/Attention-Gated-Networks
'''
import argparse
import numpy as np
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models.networks_other import adjust_learning_rate
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


# Not using anymore
def check_warm_start(epoch, model, train_opts):
    if hasattr(train_opts, "warm_start_epoch"):
        if epoch < train_opts.warm_start_epoch:
            print('... warm_start: lr={}'.format(train_opts.warm_start_lr))
            adjust_learning_rate(model.optimizers[0], train_opts.warm_start_lr)
        elif epoch == train_opts.warm_start_epoch:
            print('... warm_start ended: lr={}'.format(model.opts.lr_rate))
            adjust_learning_rate(model.optimizers[0], model.opts.lr_rate)


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
    train_dataset = ds_class(ds_path, split='train', transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='val',   transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset  = ds_class(ds_path, split='test',  transform=ds_transform['valid'], preload_data=train_opts.preloadData)

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

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    track_labels = np.arange(len(train_dataset.label_names))
    model.set_labels(track_labels)
    model.set_scheduler(train_opts)
    
    if hasattr(model, 'update_state'):
        model.update_state(0)

    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # # # --- Start ---
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure()
        # target_arr = np.zeros(14)
        # # # --- End ---

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

            # # # --- visualise distribution ---
            # for lab in labels.numpy():
            #     target_arr[lab] += 1
            # plt.clf(); plt.bar(train_dataset.label_names, target_arr); plt.pause(0.01)
            # # # --- End ---

            # Visualise predictions
            if epoch_iter <= 100:
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        pr_lbls = []
        gt_lbls = []
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            model.reset_results()
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Visualise predictions
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

                if train_opts.max_it == epoch_iter:
                    break

            # Error visualisation
            errors = model.get_accumulated_errors()
            stats = model.get_classification_stats()
            error_logger.update({**errors, **stats}, split=split)

            # save validation error
            if split == 'validation':
                valid_err = errors['CE']

        # Update the plots
        for split in ['train', 'validation', 'test']:
            # exclude bckground
            #track_labels = np.delete(track_labels, 3)
            #show_labels = train_dataset.label_names[:3] + train_dataset.label_names[4:]
            show_labels = train_dataset.label_names
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split, labels=show_labels)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        if hasattr(model, 'update_state'):
            model.update_state(epoch)

        # Update the model learning rate
        model.update_learning_rate(metric=valid_err, epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Classification Training Function')
    parser.add_argument('-c', '--config', default='./configs/config_sononet_8.json', help='training config file')
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
