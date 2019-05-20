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
import image_recover.score as score

# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='An experiment to compare the effect of image classification before and after restoration.')
parser.add_argument('-d', '--dataset', type=int, default=1, metavar='N',
                    help="choose the dataset for testing. [1: MNIST, 2:CIFAR-10] (default: 1 )")
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
else:
    print('Using CPU.')
# %% DataSet ID
datasets_id = {1: 'MNIST', 2: 'CIFAR-10'}
try:
    datasets_name = datasets_id[Args.dataset]
except:
    print('Because of a wrong dataset. You will using the default MNIST dataset.')
    datasets_name = 'MNIST'

# %% Main Function
if __name__ == '__main__':
    if datasets_name == 'MNIST':
        #%% Read Data
        print('Using the MNIST dataset.')
        data_train, data_test = datasets.MNIST(True)
        num_index = [1, 3, 5, 7, 2, 0, 13, 15, 17, 4]  # save 0,1,2,3,4,5,6,7,8,9
        axis = 2
        data = data_train.data[:1000]
        data = np.array(data)[:, np.newaxis, ...]
        # networks
        data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=Args.batch_size, shuffle=True)
        data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=Args.batch_size, shuffle=True)
        cnn0 = models.CNN_MNIST1().cuda() if Args.cuda else models.CNN_MNIST1()
        cnn1 = models.CNN_MNIST1().cuda() if Args.cuda else models.CNN_MNIST1()
        cnn2 = models.CNN_MNIST1().cuda() if Args.cuda else models.CNN_MNIST1()
    elif datasets_name == 'CIFAR-10':
        print('Using the CIFAR-10 dataset.')
        data_train, data_test = datasets.CIFAR10(True)
        num_index = [0, 1, 2, 3, 4]  # save 0,1,2,3,4,5,6,7,8,9
        axis = 2
        data = data_train.data[:1000]
        data = np.array(data).transpose(0, 3, 1, 2)
        # networks
        data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=Args.batch_size, shuffle=True)
        data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=Args.batch_size, shuffle=True)
        cnn0 = models.CNN_CIFAR10().cuda() if Args.cuda else models.CNN_CIFAR10()
        cnn1 = models.CNN_CIFAR10().cuda() if Args.cuda else models.CNN_CIFAR10()
        cnn2 = models.CNN_CIFAR10().cuda() if Args.cuda else models.CNN_CIFAR10()
    #%% Shuffle and Recover
    data_shuffled, index0 = processing.shuffle(data, axis=axis)
    index0 = np.array(index0) # change to ndarray
    best_index = [float('inf'), []]
    # direct greed
    recover = model.ImageRecover(data_shuffled)
    data_recover, index = recover.direct_greed()
    fluency = score.fluency(data_recover)
    k_coeff, k_distance = score.Kendal(index0[index], range(len(index)))
    if fluency < best_index[0]:
        best_index[1] = index
        best_index[0] = fluency
    if Args.show:
        plt.ion()
        plt.figure(1)
        plt.subplot(1, 3, 1) # show
        datasets.demo_show(data[num_index], datasets_name)
        plt.title('Original'), plt.axis('off')
        plt.subplot(1, 3, 2) # show
        datasets.demo_show(data_shuffled[num_index], datasets_name)
        plt.title('Shuffled'), plt.axis('off')
        plt.subplot(1, 3, 3) # show
        datasets.demo_show(data_recover[num_index], datasets_name)
        plt.title('Recover %d %.2f' %(fluency, k_coeff)), plt.axis('off')
    # svd greed
    if Args.show:
        plt.figure(2)
    for i in range(1, 21):
        data_recover2, index = recover.svd_greed(u_num=i)
        fluency = score.fluency(data_recover2)
        if fluency < best_index[0]:
            best_index[1] = index
            best_index[0] = fluency
        k_coeff, k_distance = score.Kendal(index0[index], range(len(index)))
        if Args.show:
            plt.subplot(4, 5, i)
            datasets.demo_show(data_recover2[num_index], datasets_name)
            plt.title('u=%d %d %.2f' % (i, fluency, k_coeff)), plt.axis('off')
    # best greed
    if Args.show:
        plt.figure(3)
        best = data_shuffled[:, :, best_index[1], :]
        datasets.demo_show(best[num_index], datasets_name)
        plt.title('best'), plt.axis('off')
        plt.draw()
        plt.show()
    #%% CNN Train
    #Train
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer0 = torch.optim.Adam(cnn0.parameters(), lr=Args.learn_rate)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=Args.learn_rate)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=Args.learn_rate)
    if Args.show:
        plt.ion()
        figure = plt.figure(4, figsize=(10, 6))
        Accuracy0 = []
        Accuracy1 = []
        Accuracy2 = []
    print('==>Start Training.')
    for epoch in range(Args.epochs):
        for step , (x, y) in enumerate(data_loader_train):
            if Args.cuda:
                x, y = x.cuda(), y.cuda()
            cnn0.train()
            output0 = cnn0(x)
            loss0 = loss_func(output0, y)
            optimizer0.zero_grad()
            loss0.backward()
            optimizer0.step()
            cnn1.train() # train model
            output1 = cnn1(x[:,:,index0,:])
            loss1 = loss_func(output1, y)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            cnn2.train() # train model
            output2 = cnn2(x[:,:,index0[best_index[1]],:])
            loss2 = loss_func(output2, y)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            if Args.cuda:
                pred0 = torch.max(output0, 1)[1].cuda().data.squeeze()
                pred1 = torch.max(output1, 1)[1].cuda().data.squeeze()
                pred2 = torch.max(output2, 1)[1].cuda().data.squeeze()
            else:
                pred0 = torch.max(output0, 1)[1].data.squeeze()
                pred1 = torch.max(output1, 1)[1].data.squeeze()
                pred2 = torch.max(output2, 1)[1].data.squeeze()
            accuracy0 = float((pred0 == y).sum()) / float(y.size(0))
            accuracy1 = float((pred1 == y).sum()) / float(y.size(0))
            accuracy2 = float((pred2 == y).sum()) / float(y.size(0))
            print('Epoch: %s |step: %s | train loss: %.2f %.2f %.2f | accuracy: %.2f %.2f %.2f'
                  % (epoch, step, loss0.data, loss1.data, loss2.data, accuracy0, accuracy1, accuracy2))

        all_y = []
        all_pred0 = []
        all_pred1 = []
        all_pred2 = []
        for step, (x, y) in enumerate(data_loader_test):
            if Args.cuda:
                x, y = x.cuda(), y.cuda()
            cnn0.eval()  # test model
            output0 = cnn0(x)
            cnn1.eval()  # test model
            output1 = cnn1(x[:,:,index0,:])
            cnn2.eval()  # test model
            output2 = cnn2(x[:, :, index0[best_index[1]], :])
            if Args.cuda:
                pred0 = torch.max(output0, 1)[1].cuda().data.squeeze()
                pred1 = torch.max(output1, 1)[1].cuda().data.squeeze()
                pred2 = torch.max(output2, 1)[1].cuda().data.squeeze()
            else:
                pred0 = torch.max(output0, 1)[1].data.squeeze()
                pred1 = torch.max(output1, 1)[1].data.squeeze()
                pred2 = torch.max(output2, 1)[1].data.squeeze()
            all_y.append(y)
            all_pred0.append(pred0)
            all_pred1.append(pred1)
            all_pred2.append(pred2)
        # evaluate
        y = torch.cat(all_y)
        pred0 = torch.cat(all_pred0)
        pred1 = torch.cat(all_pred1)
        pred2 = torch.cat(all_pred2)
        accuracy0 = float((pred0 == y).sum()) / float(y.size(0))
        accuracy1 = float((pred1 == y).sum()) / float(y.size(0))
        accuracy2 = float((pred2 == y).sum()) / float(y.size(0))
        print('Epoch: %s | test accuracy: %.2f %.2f %.2f' % (epoch, accuracy0, accuracy1, accuracy2))
        # draw
        if Args.show:
            Accuracy0.append(accuracy0)
            Accuracy1.append(accuracy1)
            Accuracy2.append(accuracy2)
            drawing.draw_result([Accuracy0, Accuracy1, Accuracy2], figure, ['Original', 'Shuffled', 'Recover'], True)
        # empty memory
        del x, y, all_y, all_pred0, all_pred1, all_pred2, output0, output1, output2
        if Args.cuda: torch.cuda.empty_cache()  # empty GPU memory
    print('==>Finish')
    if Args.show:
        plt.ioff()
        plt.show()

