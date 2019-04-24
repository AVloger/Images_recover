# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/17 23:39
@Author  : QuYue
@File    : time_series.py
@Software: PyCharm
Introduction:
"""
#%% Import Packages
import datasets
import torch
# %%
data = datasets.RobotExecution()
# %%
feature = []
target = []
target_dict = dict()
counter = 0
for i in data:
    if i['target'] in target_dict:
        target.append(target_dict[i['target']])
    else:
        target_dict[i['target']] = counter
        counter += 1
    feature.append(i['data'])
# %%




