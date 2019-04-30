# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/25 0:20
@Author  : QuYue
@File    : drawing.py
@Software: PyCharm
Introduction: Drawing the result
"""
#%% Import Packages
import matplotlib.pyplot as plt
import numpy as np
#%% Functions
def draw_result(result ,fig, title='', show=False):
    # actionly draw the result
    xaxis = list(range(len(result)))
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xaxis, result, marker='o')
    ax.grid()
    if len(title) != 0:
        ax.set_title(title)
    if show:
        ax.annotate(s=title+': %.3f' % result[-1], xy=(xaxis[-1], result[-1]), xytext=(-20, 10), textcoords='offset points')
        r = np.array(result)
        ax.annotate(s='Max: %.3f' % r.max(), xy=(r.argmax(), r.max()), xytext=(-20, -10), textcoords='offset points')
    plt.pause(0.01)

def draw_result2(result1, result2 ,fig, title=[], show=False):
    # actionly draw the result
    xaxis1 = list(range(len(result1)))
    xaxis2 = list(range(len(result2)))
    fig.clf()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(xaxis1, result1, marker='o')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(xaxis2, result2, marker='o')
    ax1.grid()
    ax2.grid()
    if len(title) == 2:
        ax1.set_title(title[0])
        ax2.set_title(title[1])
    if show:
        ax1.annotate(s=title[0]+': %.3f' % result1[-1], xy=(xaxis1[-1], result1[-1]), xytext=(-20, 10), textcoords='offset points')
        ax2.annotate(s=title[1]+': %.3f' % result2[-1], xy=(xaxis2[-1], result2[-1]), xytext=(-20, 10), textcoords='offset points')
        r1 = np.array(result1)
        r2 = np.array(result2)
        ax1.annotate(s= 'Max: %.3f' % r1.max(), xy=(r1.argmax(), r1.max()), xytext=(-20, -10), textcoords='offset points')
        ax2.annotate(s='Max: %.3f' % r2.max(), xy=(r2.argmax(), r2.max()), xytext=(-20, -10), textcoords='offset points')
    plt.pause(0.01)
#%% Main Function
if __name__ == '__main__':
    fig = plt.figure(1)
    plt.ion()
    b = []
    for i in range(100):
        a = np.random.randn(1)
        b.append(a[0])
        draw_result(b, fig, 'b', True)
    plt.ioff()
    plt.show()