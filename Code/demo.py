# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/16 4:46
@Author  : QuYue
@File    : demo.py
@Software: PyCharm
Introduction: demo for RGB picture recover
"""
# %% Import Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse

import processing
import model

# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='A demo for RGB image recover.')
parser.add_argument('-i', '--image', type=str, default='lena.jpg', metavar='str',
                    help="the name of the image. (default: 'lena.jpg' )")
parser.add_argument('-p', '--path', type=str, default='../Data/Images/', metavar='str',
                    help="the path of the picture. (default: '../Data/Images/')")
Args = parser.parse_args() # the Arguments

# %% Functions
def rbg_feature(image, axis=0):
    if len(image.shape) == 3:
        image = image.swapaxes(axis, 0)
        image_f = np.hstack([image[..., 0], image[..., 1], image[..., 2]])
        image_f = image_f.swapaxes(axis, 0)
        return image_f
    else:
        return image

def rbg_image(image_f, axis=0):
    image_f = image_f.swapaxes(axis, 0)
    L = image_f.shape[1] // 3
    image = np.array([image_f[:, :L], image_f[:, L: 2*L], image_f[:, 2*L: 3* L]])
    image= image.transpose(1, 2, 0)
    image = image.swapaxes(axis, 0)
    return image



#%% Main Function
if __name__ == '__main__':
    # %% Read Data
    image_name = Args.image # get the images name
    image_path = Args.path # get the path of the images
    image = mpimg.imread(image_path + image_name) # read the images
    image_f = rbg_feature(image)
    plt.ion()
    plt.figure(1) # show
    plt.subplot(1, 2, 1)
    score = np.sum(np.abs(np.diff(image_f, axis=0)))
    plt.imshow(rbg_image(image_f))
    plt.axis('off')
    plt.title('%s %d' %(image_name, score))
    # %% Shuffle
    shuffle_image, _ = processing.shuffle(image_f, axis=0)
    plt.subplot(1, 2, 2) # show
    plt.imshow(rbg_image(shuffle_image), cmap='Greys_r')
    score = np.sum(np.abs(np.diff(shuffle_image, axis=0)))
    plt.title('shuffle %d' %score)
    plt.axis('off')
    plt.draw()
    # %% SVD image sort
    new_image1 = model.svd_imsort(shuffle_image)
    # show
    plt.figure(2)
    plt.imshow(rbg_image(new_image1), cmap='gray')
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
        plt.imshow(rbg_image(new_image2), cmap='gray')
        score = np.sum(np.abs(np.diff(new_image2, axis=0)))
        plt.title('u=%d %d' %(i, score))
        plt.axis('off')
    plt.draw()
    # %% Direct greed
    plt.figure(4)
    new_image3, _ = recover.direct_greed()
    plt.imshow(rbg_image(new_image3), cmap='Greys_r')
    score = np.sum(np.abs(np.diff(new_image3, axis=0)))
    plt.title('Direct_greed %d' %score)
    plt.axis('off')
    plt.draw()
    plt.ioff()
    plt.show()










