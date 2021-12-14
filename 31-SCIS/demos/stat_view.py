import cv2
import os

import numpy as np

from roma import console
from lambo import DaVinci
import matplotlib.pyplot as plt



class StatViewer(DaVinci):

  def __init__(self, p: list, names: list, n_frames=100):
    # Call parent's constructor
    super().__init__('StatViewer', 3, 4)

    assert len(p) == len(names)
    self.p = p
    self.names = names

    self._init_distribution(n_frames)

    #
    self.add_plotter(self._plot_hist)


  def _init_distribution(self, N: int):
    import random

    stat = [0] * len(self.p)
    cells = random.choices(list(range(len(self.p))), weights=self.p, k=N)
    objects = [stat.copy()]
    for c in cells:
      stat[c] += 1
      objects.append(stat.copy())

    self.objects = objects


  def _plot_hist(self, x: list, ax: plt.Axes):
    index = np.arange(len(self.p)) + 0.3
    bar_width = 0.5
    S = np.sum(x)
    color = [1.0 for i in x] if S == 0 else np.array(x) / S
    ax.bar(index, x, bar_width, color=color)
    ax.set_ylabel('Cell Count')
    ax.set_ylim([0, max(self.objects[-1]) * 1.1])
    plt.xticks(list(index), self.names)

    ax.set_title('Statistic')



if __name__ == '__main__':
  cv = StatViewer(
    [0.2, 0.1, 0.4, 0.3], ['B-Cell', 'T-Cell', 'G-Cell', 'M-Cell'])
  cv.show()

