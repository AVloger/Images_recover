# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/22 20:02
@Author  : QuYue
@File    : datasets.py
@Software: PyCharm
Introduction:
"""
# %% Import Packages
import torch
import torchvision

#%% Functions
def MNIST(download=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root = "../../Data/MNIST/",
                                            transform=transform,
                                            train=True,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST/",
                                           transform=transform,
                                           train=False,
                                           download=download)
    return data_train, data_test

#%% Main Function
if __name__ == '__main__':
    data_train, data_test = MNIST()