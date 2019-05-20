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
import numpy as np
import matplotlib.pyplot as plt

#%% Functions
def MNIST(download=False):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root="../../Data/MNIST/",
                                            transform=transform,
                                            train=True,
                                            download=download)
    data_test = torchvision.datasets.MNIST(root="../../Data/MNIST/",
                                           transform=transform,
                                           train=False,
                                           download=download)
    return data_train, data_test

def CIFAR10(download=False):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = torchvision.datasets.CIFAR10(root="../../Data/CIFAR10/",
                                              transform=transform,
                                              train=True,
                                              download=download)
    data_test = torchvision.datasets.CIFAR10(root="../../Data/CIFAR10/",
                                             transform=transform,
                                             train=False,
                                             download=download)
    return data_train, data_test

def demo_show(data, dataset='MNIST'):
    if dataset == 'MNIST':
        plt.imshow(np.hstack(data[:, 0, ...]), cmap='gray')
    elif dataset == 'CIFAR-10':
        plt.imshow(np.hstack(data.transpose(0, 2, 3, 1)))
    else:
        pass


#%% Main Function
if __name__ == '__main__':
    data_train, data_test = CIFAR10(True)
