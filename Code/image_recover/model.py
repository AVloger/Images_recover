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
class ImageRecover():
    # the image recover algorithms
    def __init__(self, image):
        """
        A class for recovering images which has been shuffled.
        Parameters
        ---------
            image: ndarray
                The image for calculation fluency. The dimension of this tensor can be :
                    (4)image_num * channels * row(be shuffled) * column
                    (3)image_num * row(be shuffled) * column
                    (2)row(be shuffled) * column
        """
        image = self.check(image)
        self.image = image                   # image(image_num * channels * row(be shuffled) * column)
        self.L = self.image.shape[2]      # the length of the shuffled axis

    def check(self, image):
        if len(image.shape) == 4:
            pass
        elif len(image.shape) == 3:
            image = image[:, np.newaxis, ...]
        elif len(image.shape) == 2:
            image = image[np.newaxis, np.newaxis, ...]
        else:
            print('Please input the right image(image_num * channels * row(be shuffled) * column)')
            image = None
        return image

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

    def svd_imsort(self, seed=None):
        new_image, index = self.svd_greed(u_num=1, seed=seed)
        return new_image, index

    def svd_greed(self, u_num=3, seed=None):
        if u_num == 0:
            return self.image
        feature = []
        for img in self.image:
            img = np.hstack(img) # channels stack
            u, sigma, vt = la.svd(img)
            feature.append(u[:, :u_num])
        self.feature = np.hstack(feature)
        index = self.greed(seed)
        new_image = self.image[:, :, index, :]
        return new_image, index

    def direct_greed(self, seed=None):
        self.feature = np.hstack(np.vstack(self.image))
        index = self.greed(seed)
        new_image = self.image[:, :, index, :]
        return new_image, index


#%% Main Function
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import processing
    #%% Arguments
    path = './images/'
    file = ['YaoMing.jpg', 'Kim.jpg', 'Hanazawa.jpg']
    num = len(file)
    # %% Read images
    plt.figure(1, figsize=(num * 2, 8))
    images = []
    for i in range(num):
        img = mpimg.imread(path + file[i])
        images.append(img)
        plt.subplot(5, num, i+1)
        plt.imshow(img)
        plt.title(file[i].split('.')[0])
        plt.axis('off')
    #%% Shuffled
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)
    images, _ = processing.shuffle(images, 2)
    for i in range(num):
        img_s = images[i].transpose(1, 2, 0)
        plt.subplot(5, num, i+num+1)
        plt.imshow(img_s)
        plt.title('Shuffled')
        plt.axis('off')
    #%% SVD imsort
    recover = ImageRecover(images)
    images_r0, _ = recover.svd_imsort()
    for i in range(num):
        img_r = images_r0[i].transpose(1, 2, 0)
        plt.subplot(5, num, i+num*2+1)
        plt.imshow(img_r)
        plt.title('SVD imsort')
        plt.axis('off')
    #%% Direct greed
    recover = ImageRecover(images)
    images_r1, _ = recover.direct_greed()
    for i in range(num):
        img_r = images_r1[i].transpose(1, 2, 0)
        plt.subplot(5, num, i+num*3+1)
        plt.imshow(img_r)
        plt.title('Direct greed')
        plt.axis('off')
    #%% SVD greed
    images_r2, _ = recover.svd_greed(u_num=10)
    for i in range(num):
        img_r = images_r2[i].transpose(1, 2, 0)
        plt.subplot(5, num, i+num*4+1)
        plt.imshow(img_r)
        plt.title('SVD greed')
        plt.axis('off')
    plt.show()