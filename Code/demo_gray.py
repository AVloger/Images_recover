# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/10 10:40
@Author  : QuYue
@File    : demo_gray.py
@Software: PyCharm
Introduction: A demo for picture recover
"""
# %% Import Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse

import processing
import model

# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='A demo for gray image recover.')
parser.add_argument('-i', '--image', type=str, default='lena.jpg', metavar='str',
                    help="the name of the image. (default: 'lena.jpg' )")
parser.add_argument('-p', '--path', type=str, default='../Data/Images/', metavar='str',
                    help="the path of the picture. (default: '../Data/Images/')")
Args = parser.parse_args() # the Arguments

#%% Main Function
if __name__ == '__main__':
    # %% Read Data
    image_name = Args.image # get the images name
    image_path = Args.path # get the path of the images
    image = mpimg.imread(image_path + image_name) # read the images
    plt.ion()
    plt.figure(1) # show
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(image_name)
    # %% Gray
    gray_image = processing.rgb2gray(image)
    plt.subplot(1, 3, 2) # show
    plt.imshow(gray_image, cmap='gray')
    score = np.sum(np.abs(np.diff(gray_image, axis=0)))
    plt.axis('off')
    plt.title('%s-Gray %d' %(image_name, score))
    # %% Shuffle
    shuffle_image, _ = processing.shuffle(gray_image, axis=0)
    plt.subplot(1, 3, 3) # show
    plt.imshow(shuffle_image, cmap='Greys_r')
    score = np.sum(np.abs(np.diff(shuffle_image, axis=0)))
    plt.title('shuffle %d' %score)
    plt.axis('off')
    plt.draw()
    # %% SVD image sort
    new_image1 = model.svd_imsort(shuffle_image)
    # show
    plt.figure(2)
    plt.imshow(new_image1, cmap='gray')
    score = np.sum(np.abs(np.diff(new_image1, axis=0)))
    plt.title('SVD_imsort %d' %score)
    plt.axis('off')
    plt.draw()

    # %% SVD greed
    recover = model.ImageRecover(shuffle_image)
    plt.figure(3)
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        new_image2, _ = recover.svd_greed(u_num=i)
        plt.imshow(new_image2, cmap='gray')
        score = np.sum(np.abs(np.diff(new_image2, axis=0)))
        plt.title('u=%d %d' %(i, score))
        plt.axis('off')
    plt.draw()
    # %% Direct greed
    plt.figure(4)
    new_image3, _ = recover.direct_greed()
    plt.imshow(new_image3, cmap='Greys_r')
    score = np.sum(np.abs(np.diff(new_image3, axis=0)))
    plt.title('Direct_greed %d' %score)
    plt.axis('off')
    plt.draw()
    plt.ioff()
    plt.show()









