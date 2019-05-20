# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/20 18:10
@Author  : QuYue
@File    : demo_rc_gray.py
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
parser = argparse.ArgumentParser(description='A demo for gray image recover.')
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
    fluency = score.fluency(gray_image)
    plt.axis('off')
    plt.title('%s-Gray %d' %(image_name, fluency))

    # %% Shuffle
    shuffle_image, index0 = processing.shuffle(gray_image, axis=0)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=1)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=0)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=1)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=0)
    shuffle_image, index0 = processing.shuffle(shuffle_image, axis=1)


    plt.subplot(1, 3, 3) # show
    plt.imshow(shuffle_image, cmap='gray')
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
        new_image3, index = recover.direct_greed()
        t_image.append(new_image3[0][0])
        fluency.append(score.fluency(t_image[-1]))
        for j in range(40):
            new_image3, index = recover.svd_greed(j+1)
            t_image.append(new_image3[0][0])
            fluency.append(score.fluency(t_image[-1]))
        pos = np.array(fluency).argmin()
        print(pos)
        temp_image = t_image[pos]
        plt.subplot(2, 5, i+1)
        plt.title('greed %d' % (fluency[pos]))
        plt.imshow(temp_image, cmap='Greys_r')
        temp_image = temp_image.transpose()

        plt.axis('off')
        plt.draw()
    plt.ioff()
    plt.show()