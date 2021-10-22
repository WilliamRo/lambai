import numpy as np
import matplotlib.pyplot as plt

from lambo import DaVinci
from tframe.data.shadow import DataShadow
from pet_set import PetSet


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
    im = self.data.features[x].data
    title = f'Pawpolarity: {self.data.targets[x]}'
    self.imshow(x=im, ax=self.axes, title=title)

  def histogram(self, ax: plt.Axes, bins=101, density=True):
    data = [self.data.targets[i] for i in self.objects]
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
      self.objects = list(range(len(self.data.features)))
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

    def histogram(ax: plt.Axes, bins=101, density=True, key=None):
      # Overlap two histograms
      for v in (0, 1):
        p = [self.data.targets[i] for i in self.objects
             if self.data[key][i] == v]
        ax.hist(p, bins=bins, density=density, alpha=0.5)

      # Finalize
      ax.set_aspect('auto')
      ax.set_title('Pawpularity Histogram')
      ax.legend([f'{key} = 0', f'{key} = 1'])

    for key in self.data.meta_keys:
      self.add_plotter(lambda ax, bins=101, density=True, key=key: histogram(
        ax, bins, density, key))

  # endregion: Public Methods


if __name__ == '__main__':
  from pet_core import th
  from pet_agent import PetAgent

  ps = PetAgent.load_as_tframe_data(th.data_dir)
  ps.visualize()

