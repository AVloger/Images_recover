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
            torch.nn.Conv2d(1, 16, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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