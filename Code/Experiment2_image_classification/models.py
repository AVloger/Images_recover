# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/22 21:19
@Author  : QuYue
@File    : models.py
@Software: PyCharm
Introduction: The deep learning models for image classification
"""
#%% Import Packages
import torch

#%% Functions
class CNN_MNIST1(torch.nn.Module):
    def __init__(self):
        super(CNN_MNIST1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16 * 7 * 7, 400),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(400, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.dense(x)
        return output

class CNN_CIFAR10(torch.nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(0.5)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout(0.5)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.5)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(2048, 1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1000, 400),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(400, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.dense(x)
        return output


#%% Main Function
if __name__ == '__main__':
    import numpy as np
    cnn = CNN_MNIST1()
    x = np.random.randn(5, 1, 28, 28)
    x = torch.Tensor(x)
    y = cnn(x)
    print(y.shape)
    cnn = CNN_CIFAR10()
    x = np.random.randn(5, 3, 32, 32)
    x = torch.Tensor(x)
    y = cnn(x)
    print(y.shape)