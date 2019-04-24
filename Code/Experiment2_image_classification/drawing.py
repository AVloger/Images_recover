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