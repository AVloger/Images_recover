# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/26 17:18
@Author  : QuYue
@File    : score.py
@Software: PyCharm
Introduction: The score to evaluate the effect of the image recover.
"""
# %% Import Packages
import numpy as np
# %% Functions
def __check(image):
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

def __distance_cal(vector1, vector2, type='L1'):
    if type == 'L1':
        distance = np.linalg.norm(vector1 - vector2, ord=1)
    elif type == 'L2':
        distance = np.linalg.norm(vector1 - vector2)
    elif type == 'cos':
        distance = np.dot(vector1, vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    else:
        # print("Input a wrong Distance calculation method, will using the default method 'L1'.")
        distance = np.linalg.norm(vector1 - vector2, ord=1)
    return distance


def fluency(image, distance='L1'):
    """
    A function for evaluate images' fluency.
    Parameters
    ---------
        image: ndarray
            The image for calculation fluency. The dimension of this tensor can be :
                (4)image_num * channels * row(the axis be calculated) * column
                (3)image_num * row(the axis be calculated) * column
                (2)row(the axis be calculated) * column
        distance: string
            The method of calculating the distance.(default:'L1')
            'L1'(Manhattan Distance), 'L2'(Euclidean Distance), 'cos'(Cosine)
    Returns
    ---------
        fluency_score: float
            The fluency of the image.
    """
    # check
    image = __check(image)
    if type(image) == type(None):
        return None
    # stack the images
    image = np.hstack(np.vstack(image))
    # calculation
    fluency_score = 0.0
    for i in range(image.shape[0]-1):
        fluency_score +=  __distance_cal(image[i], image[i+1], type=distance)
    return fluency_score

def Kendal(order1, order2):
    """
    A function for calculating the Kendall rank correlation coefficient and Kendall tau rank distance between the two orders by Kendall tau distance.
    Parameters
    ---------
        order1: List or ndarray   e.g. [2, 3, 1, 4, 0]
            A order.(It must be composed of different integers)
        order2: List or ndarray   e.g. [2, 3, 0, 1, 4]
            same as order1.
    Returns
    ---------
        k_coeff: float
            The Kendall rank correlation coefficient between order1 and order2.(same=1, reverse=-1)
        k_distance: int
            The Kendall tau rank distance between order1 and order2.
    """
    order1 = np.array(order1).copy()
    order2 = np.array(order2).copy()
    # check
    if (len(order1) != len(set(order1))) or (len(order2) != len(set(order2))):
        print('An order must be composed of different integers!)')
        return None
    if set(order1) != set(order2):
        print('The elements of order1 and order2 must be same!)')
        return None
    # distance
    k_distance = 0
    k_coeff = 0
    elements = list(set(order1))
    n = len(elements)
    position_dict1 = dict(zip(order1, range(n)))
    position_dict2 = dict(zip(order2, range(n)))
    for i in range(n - 1):
        for j in range(i+1, n):
            if (position_dict1[elements[i]] < position_dict1[elements[j]]) == (position_dict2[elements[i]] < position_dict2[elements[j]]):
                k_coeff += 1
            else:
                k_distance += 1
                k_coeff += -1
    k_coeff = k_coeff / (n * (n - 1) / 2)
    return k_coeff, k_distance

 #%% Main Function
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import processing
    #%% Arguments
    path = './images/'
    file = ['YaoMing.jpg', 'Kim.jpg']
    num = len(file)
    # %% Read images
    plt.figure(1, figsize=(num * 3, 3))
    images = []
    for i in range(num):
        img = mpimg.imread(path + file[i])
        images.append(img)
    #%% fluency
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)
    for i in range(num):
        img = images[i].transpose(1, 2, 0)
        fluency_score = fluency(img)
        plt.subplot(1, num, i+1)
        plt.imshow(img)
        plt.title(file[i].split('.')[0] + ' fluency:%d' %fluency_score)
        plt.axis('off')
    plt.show()
    #%% Kendall tau distance
    order1 = [0, 3, 1, 6, 2, 5, 4]
    order2 = [1, 0, 3, 6, 4, 2, 5]
    k_coeff, k_distance = Kendal(order1, order2)

