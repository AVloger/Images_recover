# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/20 16:44
@Author  : QuYue
@File    : demo_rc.py
@Software: PyCharm
Introduction:
"""
# %% Import Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import sys
sys.path.append('..') # add the path which includes the packages
import image_recover.processing as processing
import image_recover.model as model
import image_recover.score as score

# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='A demo for RGB image recover.')
parser.add_argument('-i', '--image', type=str, default='lena.jpg', metavar='str',
                    help="the name of the image. (default: 'lena.jpg' )")
parser.add_argument('-p', '--path', type=str, default='../../Data/Images/', metavar='str',
                    help="the path of the picture. (default: '../../Data/Images/')")
Args = parser.parse_args() # the Arguments

#%% Main Function
if __name__ == '__main__':
    # %% Read Data
    image_name = Args.image # get the images name
    image_path = Args.path # get the path of the images
    image = mpimg.imread(image_path + image_name) # read the images
    # change the axis
    image_f = image.transpose(2, 0, 1)
    image_f = image_f[np.newaxis, ...]
    # show
    plt.ion()
    plt.figure(1) # show
    plt.subplot(1, 2, 1)
    fluency = score.fluency(image_f)
    plt.imshow(image_f[0].transpose(1, 2, 0))
    plt.axis('off')
    plt.title('%s %d' %(image_name, fluency))

    # # %% Shuffle
    # shuffle_image, index0  = processing.shuffle(image_f, axis=2)
    # index0 = np.array(index0) # change to ndarray
    # plt.subplot(1, 2, 2) # show
    # plt.imshow(shuffle_image[0].transpose(1, 2, 0))
    # fluency = score.fluency(shuffle_image)
    # plt.title('shuffle %d' %fluency)
    # plt.axis('off')
    # plt.draw()
    # %% Shuffle
    shuffle_image, index0 = processing.shuffle(image_f, axis=2)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=3)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=2)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=3)


    plt.subplot(1, 2, 2) # show
    plt.imshow(shuffle_image[0].transpose(1, 2, 0))
    fluency = score.fluency(shuffle_image)
    plt.title('shuffle %d' %fluency)
    plt.axis('off')
    plt.draw()

    #%%
    plt.figure(2)
    temp_image = shuffle_image
    for i in range(10):
        recover = model.ImageRecover(temp_image)
        t_image = []
        fluency=[]
        new_image, index = recover.direct_greed()
        t_image.append(new_image)
        fluency.append(score.fluency(t_image[-1]))
        for j in range(30):
            new_image, index = recover.svd_greed(j+1)
            t_image.append(new_image)
            fluency.append(score.fluency(t_image[-1]))
        pos = np.array(fluency).argmin()
        print(pos)
        temp_image = t_image[pos]
        plt.subplot(2, 5, i+1)
        plt.title('greed %d' % (fluency[pos]))
        plt.imshow(temp_image[0].transpose(1, 2, 0))
        temp_image = temp_image.swapaxes(2, 3)

        plt.axis('off')
        plt.draw()
    plt.ioff()
    plt.show()