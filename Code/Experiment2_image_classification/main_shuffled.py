# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/25 15:26
@Author  : QuYue
@File    : main_shuffled.py
@Software: PyCharm
Introduction: the main function for image classification(shuffled).
"""
# %% Import Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
import argparse
import datasets
import models
import drawing
import sys
sys.path.append('..') # add the path which includes the packages
import image_recover.processing as processing
import image_recover.model as model

# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='An experiment to compare the effect of image classification before and after restoration.')
parser.add_argument('-d', '--dataset', type=int, default=1, metavar='N',
                    help="choose the dataset for testing. [1: MNIST] (default: 1 )")
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('-b', '--batch-size', type=int, default=10, metavar='N',
                    help="the batch size for the training and test (default: 10 )")
parser.add_argument('-l', '--learn-rate', type=float, default=0.001, metavar='Float',
                    help='learning-rate for training(default: 0.001)')
parser.add_argument('-c', '--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-s', '--show', action='store_true', default=False,
                    help='enables show the accuracy')

Args = parser.parse_args() # the Arguments
# Args.cuda = True
# Args.show = True
if Args.cuda:
    print('Using GPU.')
# %% DataSet ID
datasets_id = {1: 'MNIST'}
try:
    datasets_name = datasets_id[Args.dataset]
except:
    print('Because of a wrong dataset. You will using the default MNIST dataset.')
    datasets_name = 'MNIST'
#%%
def images_feature(image, axis=0, axis_num=2):
    # axis: the shuffled axis
    # axis_num: the num axis
    if len(image.shape) == 3 and axis != axis_num:
        image = image.transpose(axis,2,0)
        temp = []
        for i in range(image)
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
# %% Main Function
if __name__ == '__main__':
    if datasets_name == 'MNIST':
        print('Using the MNIST dataset.')
        data_train, data_test = datasets.MNIST(True)
        data = data_train.data[:3]
        axis = 0
        data = np.array(data).transpose(1,2,0)
        data = rbg_feature(data)
        data_shuffled, _ = processing.shuffle(data, axis=axis)
        recover = model.ImageRecover(data_shuffled, axis=axis)
        data_recover,_ = recover.direct_greed()
        plt.figure(1)
        plt.subplot(1, 3, 1) # show
        plt.imshow(data, cmap='gray')
        plt.title('Original')
        plt.subplot(1, 3, 2) # show
        plt.imshow(data_shuffled, cmap='gray')
        plt.title('Shuffled')
        plt.subplot(1, 3, 3) # show
        plt.imshow(data_recover, cmap='gray')
        plt.title('Recover')
        #%%
        plt.figure(2)
        for i in range(1, 21):
            plt.subplot(4, 5, i)
            new_image2, _ = recover.svd_greed(u_num=i)
            plt.imshow(new_image2, cmap='gray')
            score = np.sum(np.abs(np.diff(new_image2, axis=0)))
            plt.title('u=%d %d' % (i, score))
            plt.axis('off')
        plt.show()




        # data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=Args.batch_size, shuffle=True)
        # data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=Args.batch_size, shuffle=True)

        # cnn = models.CNN_MNIST1().cuda() if Args.cuda else models.CNN_MNIST1()
        # loss_func = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(cnn.parameters(), lr=Args.learn_rate)
        # if Args.show:
        #     plt.ion()
        #     figure = plt.figure(1)
        #     Accuracy = []
        # print('==>Start Training.')
        # for epoch in range(Args.epochs):
        #     for step , (x, y) in enumerate(data_loader_train):
        #         if Args.cuda:
        #             x, y = x.cuda(), y.cuda()
        #         cnn.train() # train model
        #         output = cnn(x)
        #         loss = loss_func(output, y)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         if Args.cuda:
        #             pred = torch.max(output, 1)[1].cuda().data.squeeze()
        #         else:
        #             pred = torch.max(output, 1)[1].data.squeeze()
        #         accuracy = float((pred == y).sum()) / float(y.size(0))
        #         print('Epoch: %s |step: %s | train loss: %.2f | accuracy: %.2f' % (epoch, step, loss.data, accuracy))
        #
        #     all_y = []
        #     all_pred = []
        #     for step, (x, y) in enumerate(data_loader_test):
        #         if Args.cuda:
        #             x, y = x.cuda(), y.cuda()
        #         cnn.eval()  # test model
        #         output = cnn(x)
        #         if Args.cuda:
        #             pred = torch.max(output, 1)[1].cuda().data.squeeze()
        #         else:
        #             pred = torch.max(output, 1)[1].data.squeeze()
        #         all_y.append(y)
        #         all_pred.append(pred)
        #     # evaluate
        #     y = torch.cat(all_y)
        #     pred = torch.cat(all_pred)
        #     accuracy = float((pred == y).sum()) / float(y.size(0))
        #     print('Epoch: %s | test accuracy: %.2f' % (epoch, accuracy))
        #     # draw
        #     if Args.show:
        #         Accuracy.append(accuracy)
        #         drawing.draw_result(Accuracy, figure, 'Accuracy', True)
        #     # empty memory
        #     del x, y, all_y, all_pred, output
        #     if Args.cuda: torch.cuda.empty_cache()  # empty GPU memory
        # print('==>Finish')
        # if Args.show:
        #     plt.ioff()
        #     plt.show()

