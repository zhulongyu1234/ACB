import argparse
import random
from datetime import datetime
import torch.utils.data as data

from datasets_operations import *
from utils import *

from train_test import *
from sklearn.metrics import classification_report
from results_report import metrics

import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
from ACB import ACB

datasets_file = {
    'PU': ['paviaU.mat', 'paviaU_7gt.mat'],
    'PC': ['paviaC.mat', 'paviaC_7gt.mat'],
    'D': ['Dioni.mat', 'Dioni_gt_out68.mat'],
    'L': ['Loukia.mat', 'Loukia_gt_out68.mat'],
    'H13': ['Houston13.mat', 'Houston13_7gt.mat'],
    'H18': ['Houston18.mat', 'Houston18_7gt.mat'],
}


parser = argparse.ArgumentParser(description='important parameters')
parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='../datasets_preprocess/DATASETS/',
                    help='the path to load the data')
parser.add_argument('--model_results', type=str, default="./ACB/",
                    help='the path to save the model')

parser.add_argument('--pretrained_model_path', type=str, default=None,
                    help='If there is a pre-trained model, this is set to its path')
parser.add_argument('--cuda', type=int, default=-1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")  # CPU:-1 || GPU:0

parser.add_argument('--dataset_name', type=str, default='Houston',
                    help='Task dataset name')
source_data, source_label = datasets_file['H13']
target_data, target_label = datasets_file['H18']
parser.add_argument('--source_data', type=str, default=source_data,
                    help='the name of the source data file')
parser.add_argument('--source_label', type=str, default=source_label,
                    help='the name of the source label file')
parser.add_argument('--target_data', type=str, default=target_data,
                    help='the name of the test data file')
parser.add_argument('--target_label', type=str, default=target_label,
                    help='the name of the test label file')

parser.add_argument('--patch_size', type=int, default=12, help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate, set by the model if not specified.")
parser.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")
parser.add_argument('--seed', type=int, default=555, metavar='S',
                    help='random seed ')

parser.add_argument('--log_interval', type=int, default=4, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_epoch', type=int, default=300,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.4,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=3,
                    help='multiple of of data augmentation')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")


args = parser.parse_args()
DEVICE = get_device(args.cuda)

if __name__ == '__main__':
    seed_worker(args.seed)  # args.seed

    source_hsi, source_gt, target_hsi, target_gt, ignored_labels = get_dataset(args)

    sample_num_src = len(np.nonzero(source_gt)[0])
    sample_num_tar = len(np.nonzero(target_gt)[0])

    num_classes = int(source_gt.max())
    N_BANDS = source_hsi.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full', 'seed': args.seed})

    r = int(hyperparams['patch_size'] / 2)

    source_hsi = np.pad(source_hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    target_hsi = np.pad(target_hsi, ((r, r), (r, r), (0, 0)), 'symmetric')

    source_gt = np.pad(source_gt, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    target_gt = np.pad(target_gt, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, _, _, _ = sample_gt(source_gt, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(target_gt, 1, mode='random')
    source_hsi_re, train_gt_src_re = source_hsi, train_gt_src

    for i in range(args.re_ratio - 1):
        source_hsi_re = np.concatenate((source_hsi_re, source_hsi))
        train_gt_src_re = np.concatenate(
            (train_gt_src_re, train_gt_src))

    train_dataset = HyperX(source_hsi_re, train_gt_src_re, **hyperparams)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True,
                                   drop_last=True)
    test_dataset = HyperX(target_hsi, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  # worker_init_fn=seed_worker,
                                  # generator=g,
                                  batch_size=hyperparams['batch_size'],
                                  drop_last=False)
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    hyperparams.update({'Number of training set samples': len_src_dataset, 'Number of testing set samples': len_tar_dataset})

    model = ACB(num_classes, N_BANDS, hyperparams['patch_size']).to(DEVICE)
    model_dict = model.state_dict()
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        pretrained_dict = torch.load(args.pretrained_model_path, map_location=DEVICE).state_dict()
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    root_path = os.path.join(args.save_path)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    task_logit_dir = os.path.join(root_path, args.dataset_name, args.model_results, time_str)
    if not os.path.exists(task_logit_dir):
        os.makedirs(task_logit_dir)
    with open(os.path.join(task_logit_dir, 'hyperparams.txt'), 'a') as file:
        for k, v in hyperparams.items():
            print(f'{k}:{v}')
            file.write(f'{k}:{v}\n')
        file.write(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.\n')

    correct, max_acc = 0, 0

    test_accuracy, predict_list, label_list = test(model, test_loader, DEVICE, task_logit_dir)
    for epoch in range(1, args.num_epoch + 1):
        model, _ = train(args, epoch, train_loader, len_src_dataset, DEVICE, model)
        test_accuracy, predict_list, label_list = test(model, test_loader, DEVICE, task_logit_dir)

        if test_accuracy > max_acc:
            max_acc = test_accuracy
            results = metrics(np.concatenate(predict_list), np.concatenate(label_list), ignored_labels=hyperparams[
                'ignored_labels'], n_classes=num_classes)

            class_indexes = []
            for class_index in range(1, num_classes + 1):
                class_indexes.append(str(class_index))
            logs_report = classification_report(np.concatenate(predict_list), np.concatenate(label_list), target_names=class_indexes)
            print(logs_report)
            logs_max = 'max accuracy{: .2f}%\n'.format(max_acc)
            print(logs_max, end='')
            with open(os.path.join(task_logit_dir, 'hyperparams.txt'), 'a') as file:
                print(f'model_epoch{epoch}_acc{test_accuracy}')
                file.write(f'model_epoch{epoch}_acc{test_accuracy}\n')

                # torch.save(model, os.path.join(task_logit_dir, 'best_model.pt'))

        else:
            logs_acc = 'Test result decline, present accuracy{: .2f}% | max accuracy{: .2f}%\n'.format(test_accuracy, max_acc)
            print(logs_acc)


