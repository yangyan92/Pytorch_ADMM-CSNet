# encoding:utf-8
import torch.utils.data as data
import os
import os.path
from scipy.io import loadmat
import re
import numpy as np
import torch
import h5py


class DataSet(data.Dataset):

    def __init__(self, dir):
        imgs = os.listdir(dir)
        imgs.sort()
        self.imgs = [os.path.join(dir, img) for img in imgs]
        self.num = [re.sub("\D", "", img) for img in imgs]

    def __getitem__(self, index):
        num = self.num[index]
        img_path = self.imgs[index]
        data = loadmat(img_path)

        data_label = data['data'][0][0][0][0][0][0]
        label_real = torch.Tensor(data_label.real)
        label_real = label_real.view(1, 256, 256)
        label_imag = torch.Tensor(data_label.imag)
        label_imag = label_imag.view(1, 256, 256)

        label = torch.complex(label_real, label_imag)
        return label, num

    def __len__(self):
        return len(self.imgs)


def get_data(load_root):

    train = load_root + 'train'
    test = load_root + 'test'
    validate = load_root + 'validate'
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data


def get_test_data(load_root):

    test = load_root + 'test'
    test_data = DataSet(test)
    return test_data