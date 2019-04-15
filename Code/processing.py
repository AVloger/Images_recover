# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/12 19:41
@Author  : QuYue
@File    : processing.py
@Software: PyCharm
Introduction: Processing the images
"""
#%% Import Packages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
#%% Functions
def rgb2gray(rgb):
    # Get the gray picture from the RGB picture
    # Gray = R*0.299 + G*0.587 + B*0.114
    return np.dot(rgb, [0.299, 0.587, 0.114])

def shuffle(image, axis=0, seed=None):
    # shuffle the image
    np.random.seed(seed=seed) # set the random seed
    index = np.arange(image.shape[axis])
    np.random.shuffle(index)
    # axis change
    image = image.swapaxes(axis, 0) # change the choosen axis to the first axis
    new_image = image[index, ...]   # shuffle the choosen axis
    new_image = new_image.swapaxes(axis, 0) # change the axis back
    return new_image, index

#%% Main Function
if __name__ == '__main__':
    im_name = 'lena.jpg'
    im_path = '../Data/Images/'
    image = mpimg.imread(im_path + im_name) # read the images
    # image = rgb2gray(image)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(im_name)
    plt.axis('off')
    # shuffle
    image, _ = shuffle(image, axis=0)
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.title('shuffle')
    plt.axis('off')
    plt.show()


