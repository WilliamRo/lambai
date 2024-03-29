import numpy as np
import matplotlib.pyplot as plt

from lambo import DaVinci
from tframe.data.shadow import DataShadow
from pet.pet_set import PetSet


class PetViewer(DaVinci):

  def __init__(self, data: PetSet):
    super(PetViewer, self).__init__('Pet Viewer')
    self.data: PetSet = data

    # Initialize
    self.filter()

    # Add plotter
    self.add_plotter(self.show_pet)
    self.add_plotter(self.histogram)

  # region: Plotters

  def show_pet(self, x: int):
    im = self.data[x].features[0]
    title = (f'[{im.shape[0]}x{im.shape[1]}]' 
             f' Pawpplarity: {self.data.targets[x]}')
    self.imshow(x=im, ax=self.axes, title=title)

  def histogram(self, ax: plt.Axes, bins=100, density=True):
    data = [self.data.targets[i] for i in self.objects]
    data = np.array(data).flatten()
    ax.hist(data, bins=bins, density=density, alpha=0.7)
    ax.set_aspect('auto')
    ax.set_title('Pawpularity Histogram')

  # endregion: Plotters

  # region: Commands

  def filter(self, key=None, v1: int = None, v2: int = None):
    self._filter(key, v1, v2)
    self.refresh()

  def _filter(self, key, v1, v2):
    if key is None:
      self.objects = list(range(self.data.size))
      return

    if key in ('p', 'pawpularity', 'score'):
      self.objects = [
        i for i in self.objects if v1 <= self.data.targets[i] <= v2]
      return

    if not key in self.data.meta_keys:
      raise KeyError(f'Unknown metadata key: {key}')
    assert v1 in (0, 1)
    self.objects = [i for i in self.objects
                    if self.data.data_dict[key][i] == v1]

  # endregion: Commands

  # region: Public Methods

  def compare_hist(self):

    def histogram(ax: plt.Axes, bins=100, density=True, key=None):
      # Overlap two histograms
      for v in (0, 1):
        p = [self.data.targets[i] for i in self.objects
             if self.data[key][i] == v]
        p = np.array(p).flatten()
        ax.hist(p, bins=bins, density=density, alpha=0.5)

      # Finalize
      ax.set_aspect('auto')
      ax.set_title('Pawpularity Histogram')
      ax.legend([f'{key} = 0', f'{key} = 1'])

    for key in self.data.meta_keys:
      self.add_plotter(lambda ax, bins=100, density=True, key=key: histogram(
        ax, bins, density, key))

  def show_size_hist(self):
    assert self.data.sizes is not None

    def size_hist(ax: plt.Axes, density=True):
      heights = [s[0] for s in self.data.sizes]
      widths = [s[1] for s in self.data.sizes]
      ax.hist(heights, density=density, alpha=0.5)
      ax.hist(widths, density=density, alpha=0.5)

      # Finalize
      ax.set_aspect('auto')
      ax.set_title('Size Distribution')
      ax.legend(['Height', 'Width'])

    self.add_plotter(size_hist)

  # endregion: Public Methods


if __name__ == '__main__':
  from pet_core import th
  from pet_agent import PetAgent

  ps = PetAgent.load_as_tframe_data(th.data_dir)
  ps.visualize()

