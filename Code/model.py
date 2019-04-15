# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/13 14:38
@Author  : QuYue
@File    : model.py
@Software: PyCharm
Introduction: the models for picture recover
"""
#%% Import Packages
import numpy as np
from numpy import linalg as la
#%% Models
def svd_imsort(image, axis=0):
    # the svd imsort algorithm
    image = image.swapaxes(axis, 0) # change the choosen axis to the first axis
    u, sigma, vt = la.svd(image)
    index = np.argsort(u[:, 0])
    new_image = image[index, :]
    new_image = new_image.swapaxes(axis, 0) # change the axis back
    return new_image

class ImageRecover():
    # the image recover algorithms
    def __init__(self, image, axis=0):
        self.image = image.swapaxes(axis, 0) # image(which the first axis has been shuffled)
        self.L = self.image.shape[axis]      # the length of the shuffled axis
        self.axis = axis                     # the shuffled axis

    def distance_cal(self, vector, matrix):
        # calculate the distance between the vector to every rows from the matrix (vector:(n,), matrix:(i, n))
        if len(matrix.shape) == 1:         # if matrix is a vector
            matrix = matrix[np.newaxis, :] # add a axis
        distance = np.sum((vector - matrix) ** 2, axis=1)
        return distance

    def update(self, flag='first'):
        # update the memory
        if len(self.unused) == 0: # if 'unused' has no element（the recover is over）
            return None
        if flag == 'first': # find the closest row with the first row from the 'unused'
            dis = self.distance_cal(self.feature[self.used[0], :], self.feature[self.unused, :])
            ind = np.argmin(dis)
            self.memory[0], self.memory[2] = self.unused[ind], dis[ind] # update the memory
        elif flag == 'last': # find the closest row with the last row from the 'unused'
            dis = self.distance_cal(self.feature[self.used[-1], :], self.feature[self.unused, :])
            ind = np.argmin(dis)
            self.memory[1], self.memory[3] = self.unused[ind], dis[ind]  # update the memory
        else:
            print('flag = first or last')

    def greed(self, seed=None):
        np.random.seed(seed=seed)  # set the random seed
        self.used = []                    # used row
        self.unused = list(range(self.L)) # unused row
        first = np.random.randint(self.L) # choose one row randomly
        self.used.append(first)
        self.unused.remove(first)
        self.memory = [1, 1, 1, 1] # the memory is [the index of the closest row with the first ，the index of the closest row with the last，
                                   # the distance between the closest row to the first， the distance between the closest row to the last]
        self.update('first')
        self.update('last')
        first, last = self.used[0], self.used[-1] # the first row, the last row
        for i in range(self.L - 1):
            if self.memory[2] < self.memory[3]: # update the first row
                self.used.insert(0, self.memory[0])
                self.unused.remove(self.memory[0])
                self.update('first')
                first = self.used[0]
                if first == self.memory[1]:
                    self.update('last')
            else:
                self.used.append(self.memory[1]) # update the last row
                self.unused.remove(self.memory[1])
                self.update('last')
                last = self.used[-1]
                if last == self.memory[0]:
                    self.update('first')
        return self.used

    def svd_greed(self, u_num=3, seed=None):
        if u_num == 0:
            return self.image.swapaxes(self.axis, 0)
        u, sigma, vt = la.svd(self.image)
        self.feature = u[:, :u_num]
        index = self.greed(seed)
        new_image = self.image[index, ...]
        new_image = new_image.swapaxes(self.axis, 0)
        return new_image, index
    def direct_greed(self, seed=None):
        self.feature = self.image
        index = self.greed(seed)
        new_image = self.image[index, ...]
        new_image = new_image.swapaxes(self.axis, 0)
        return new_image, index







#%% Main Function