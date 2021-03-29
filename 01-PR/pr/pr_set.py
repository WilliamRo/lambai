from typing import Union
from collections import OrderedDict

from lambo.data_obj.interferogram import Interferogram
from lambo.maths.random.window import random_window
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
    horizontal_list = ['targets', 'features']
    if 'predicted' in self.data_dict:
      horizontal_list.append('predicted')
    viewer = ImageViewer(
      ds, horizontal_list=horizontal_list, color_map='gist_earth')
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

  # region: APIs

  def evaluate_model(self, model):
    from tframe import Predictor
    assert isinstance(model, Predictor)
    console.show_status('Predicting {} ...'.format(self.name))
    y = model.predict(self, batch_size=1, verbose=True)
    self.data_dict['predicted'] = y
    self.view()

  @staticmethod
  def random_window(batch: DataSet, is_training: bool,
                    shape: Union[tuple, list]):
    if not is_training: return batch
    h, w = shape

    full_shape = batch.features.shape[1:3]
    features, targets = [], []
    for x, y in zip(batch.features, batch.targets):
      i, j = random_window(shape, full_shape)
      features.append(x[i:i+h, j:j+w])
      targets.append(y[i:i+h, j:j+w])

    # Set features and targets back
    batch.features = np.stack(features, axis=0)
    batch.targets = np.stack(targets, axis=0)

    return batch

  @staticmethod
  def random_window_preprocessor(shape: Union[tuple, list]):
    return lambda batch, is_training: PhaseSet.random_window(
      batch, is_training, shape)

  # endregion: APIs

