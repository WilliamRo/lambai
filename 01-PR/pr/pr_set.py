from collections import OrderedDict

from lambo.data_obj.interferogram import Interferogram
from tframe import console
from tframe.data.dataset import DataSet

import numpy as np


class PhaseSet(DataSet):
  """A dataset designed typically for phase retrieval related applications"""

  # region: Properties

  @property
  def inter_rep(self) -> Interferogram:
    return self['interferograms'][0]

  @property
  def groups(self) -> OrderedDict:
    def _init_groups():
      od = OrderedDict()
      for i, ig in enumerate(self['interferograms']):
        assert isinstance(ig, Interferogram)
        key = ig.sample_token
        # Initialize if necessary
        if key not in od: od[key] = []
        # Put index into od
        od[key].append(i)
      return od
    return self.get_from_pocket('groups', initializer=_init_groups)

  # endregion: Properties

  # region: Private Methods

  def _check_data(self):
    # Make sure data_dict is a non-empty dictionary
    if not isinstance(self.data_dict, dict) or len(self.data_dict) == 0:
      raise TypeError('!! data_dict must be a non-empty dictionary')

    data_length = len(list(self.data_dict.values())[0])

    # Check each item in data_dict
    for name, array in self.data_dict.items():
      # Check type and length
      if len(array) != data_length: raise ValueError(
        '!! {} should be of length {}'.format(name, data_length))

  # endregion: Private Methods

  # region: Public Methods

  def add_channel(self):
    if len(self.features.shape) == 3:
      assert len(self.targets.shape) == 3
      self.features = np.expand_dims(self.features, axis=-1)
      self.targets = np.expand_dims(self.targets, axis=-1)

  def view(self):
    from tframe.data.images.image_viewer import ImageViewer
    ds = self[:]
    # Process targets
    for i, y in enumerate(self.targets):
      y = y - np.min(y)
      self.targets[i] = y / np.max(y)
    viewer = ImageViewer(ds, horizontal_list=['targets', 'features'],
                         color_map='gist_earth')
    viewer.show()

  def report_data_details(self):
    console.show_info('Detail of {}'.format(self.name))
    console.supplement('Totally {} interferograms, including'.format(self.size))
    for i, key in enumerate(self.groups.keys()):
      console.supplement('[{}] {} `{}` samples'.format(
        i + 1, len(self.groups[key]), key), level=2)
    console.supplement('Image size: {}, Radius: {}'.format(
      'x'.join([str(l) for l in self.inter_rep.size]), self.inter_rep.radius))

  def get_subset_by_sample_indices(self, sample_indices, name=None):
    # Sanity check
    if isinstance(sample_indices, int): sample_indices = sample_indices,

    indices = []
    for key in [list(self.groups.keys())[si - 1] for si in sample_indices]:
      indices.extend(self.groups[key])
    subset = self[indices]

    # Set name if necessary
    if name: subset.name = name
    return subset

  # endregion: Public Methods
