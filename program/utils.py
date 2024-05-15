import torch
import numpy as np
import os
from scipy import io
import h5py


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def open_file(dataset_path, args):
    if args.dataset_name == 'Houston':
        source_hsi = np.transpose(np.array(h5py.File(os.path.join(dataset_path, args.source_data), 'r').get('ori_data')),(1,2,0))
        source_gt = np.array(h5py.File(os.path.join(dataset_path, args.source_label), 'r').get('map'))
        target_hsi = np.transpose(np.array(h5py.File(os.path.join(dataset_path, args.target_data), 'r').get('ori_data')),(1,2,0))
        target_gt = np.array(h5py.File(os.path.join(dataset_path, args.target_label), 'r').get('map'))
    else:
        source_hsi = io.loadmat(os.path.join(dataset_path, args.source_data))['ori_data']
        source_gt = io.loadmat(os.path.join(dataset_path, args.source_label))['map']
        target_hsi = io.loadmat(os.path.join(dataset_path, args.target_data))['ori_data']
        target_gt = io.loadmat(os.path.join(dataset_path, args.target_label))['map']
    return source_hsi, source_gt, target_hsi, target_gt


def hsi_preprocess(hsi, gt):
    # Filter NaN out
    nan_mask = np.isnan(hsi.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. "
              "Learning on NaN data is disabled.")
    hsi[nan_mask] = 0
    gt[nan_mask] = 0

    # Normalization
    hsi = np.asarray(hsi, dtype='float32')
    m, n, d = hsi.shape[0], hsi.shape[1], hsi.shape[2]
    hsi = hsi.reshape((m * n, -1))
    hsi = hsi / hsi.max()

    point_feature = np.sqrt(np.asarray((hsi ** 2).sum(1)))
    point_feature = np.expand_dims(point_feature, axis=1)
    point_feature = point_feature.repeat(d, axis=1)
    point_feature[point_feature == 0] = 1
    hsi = hsi / point_feature

    hsi = np.reshape(hsi, (m, n, -1))
    return hsi, gt



