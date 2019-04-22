# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/18 0:32
@Author  : QuYue
@File    : datasets.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import numpy as np
#%% Functions
def RobotExecution(path='../Data/Robot Execution Failures/', file='lp1.data'):
    # Read the Robot Execution Failures dataset
    # Read data
    save = []
    with open(path + file) as f:
        for i in f:
            save.append(i)
    # Data Extract
    data = []
    count = 0
    for i in save:
        if i[0].isalpha():
            temp = dict()
            temp['target'] = i[:-1]
            temp['data'] = []
        elif i[0] == '\t':
            temp['data'].append(i.strip().split('\t'))
        elif i[0] == '\n' and count == 0:
            temp['data'] = np.array(temp['data'], dtype=np.int)
            data.append(temp)
            count += 1
        elif i[0] == '\n' and count == 1:
            count = 0
        else:
            print('error')
    temp['data'] = np.array(temp['data'], dtype=np.int)
    data.append(temp)
    return data
#%% Main Functions
if __name__ == '__main__':
    data = RobotExecution()
