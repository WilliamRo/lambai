from typing import Union, List
from collections import OrderedDict

from lambo.data_obj.interferogram import Interferogram
from lambo.maths.random.window import random_window
from tframe import console
from tframe.data.dataset import DataSet
from typing import Optional

import re
import numpy as np


class PhaseSet(DataSet):
  """A dataset designed typically for phase retrieval related applications"""

  PREDICTED_KEY = 'predicted'
  DELTA_KEY = 'delta'
  SLOPE = 'slope'
  FLATNESS = 'flatness'
  EVAL_DETAILS = 'eval-details'
  WMAE = 'wmae'

  # region: Properties

  @property
  def inter_rep(self) -> Interferogram:
    return self['interferograms'][0]

  @property
  def groups(self) -> OrderedDict:
    def _init_groups():
      od = OrderedDict()
      for i, ig in enumerate(self.interferograms):
        assert isinstance(ig, Interferogram)
        key = ig.sample_token
        # Initialize if necessary
        if key not in od: od[key] = []
        # Put index into od
        od[key].append(i)
      return od
    return self.get_from_pocket('groups', initializer=_init_groups)

  @property
  def interferograms(self) -> List[Interferogram]:
    return self['interferograms']

  @property
  def details(self) -> list:
    if all([self.FLATNESS in self.properties, self.SLOPE in self.properties]):
      dtls = ['r = {:.1f}, s = {:.1f}'.format(r, s)
                 for r, s in zip(self[self.FLATNESS], self[self.SLOPE])]
    else: dtls = ['r = ?, s = ?' for _ in range(self.size)]
    # Append evaluation details if existing
    if self.EVAL_DETAILS in self.properties:
      e_dtls = self.properties[self.EVAL_DETAILS]
      dtls = [d + ' | ' + ed for d, ed in zip(dtls, e_dtls)]
    return dtls

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

  def normalize_features(self):
    x = self.features
    assert len(x.shape) == 3

    shape = [len(x), 1, 1]
    mu = np.mean(x, axis=(1, 2)).reshape(shape)
    sigma = np.std(x, axis=(1, 2)).reshape(shape)
    # Set back features
    self.features = (x - mu) / sigma

    # TODO: test
    # from sklearn import preprocessing
    # for i in range(self.size):
    #   x[i] = preprocessing.StandardScaler().fit_transform(x[i])
    # self.features = x

  def squash_target(self, truncate_at=12.0):
    assert truncate_at > 0
    assert np.min(self.targets) == 0
    self.targets[self.targets > truncate_at] = truncate_at
    self.targets = self.targets / truncate_at

  def view(self):
    from tframe.data.images.image_viewer import ImageViewer

    ds = self[:]
    # Process targets
    for i, y in enumerate(self.targets):
      # y = y - np.min(y)
      y_max = np.max(y)
      self.targets[i] = y / y_max
      # Rescale predicted results if necessary
      if self.PREDICTED_KEY in self.data_dict:
        pred = self.data_dict[self.PREDICTED_KEY][i]
        self.data_dict[self.PREDICTED_KEY][i] = pred / np.max(pred)
      if self.DELTA_KEY in self.data_dict:
        delta = self.data_dict[self.DELTA_KEY][i]
        self.data_dict[self.DELTA_KEY][i] = delta / np.max(delta)

    horizontal_list = ['targets', 'features']
    if self.DELTA_KEY in self.data_dict:
      horizontal_list.append(self.DELTA_KEY)
    if self.PREDICTED_KEY in self.data_dict:
      horizontal_list.append(self.PREDICTED_KEY)
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

  def get_subset_by_config_str(
      self, config_str: str, name: Optional[str] = None):
    """Format of config_str: ^-?\d*[at]-?\d*$
    For example, 'a', 'a3', '-5t', '2t3'
    """
    # Check format of config_str
    r = re.match(r'^-?\d*([at])-?\d*$', config_str)
    assert r is not None

    # Get begin and end
    sep = r.group(1)
    begin, end = config_str.split(sep)
    begin = int(begin) if begin else 0

    # Get subset
    if sep == 't':
      # Case (1) get samples from each trial
      indices = []
      for g in self.groups.values():
        _end = int(end) if end else len(g)
        indices.extend(g[begin:_end])
      subset = self[indices]
    else:
      assert sep == 'a'
      # Case (2) otherwise get samples for whole set
      end = int(end) if end else self.size
      subset = self[begin:end]

    # Set name if necessary
    if name: subset.name = name
    else: subset.name = self.name + '({})'.format(config_str)
    return subset

  def get_subset_by_trial_indices(self, sample_indices, name=None):
    # Sanity check
    if isinstance(sample_indices, int): sample_indices = sample_indices,

    indices = []
    for key in [list(self.groups.keys())[si - 1] for si in sample_indices]:
      indices.extend(self.groups[key])
    subset = self[indices]

    # Set name if necessary
    if name: subset.name = name
    return subset

  def show_hist(self):
    import matplotlib.pyplot as plt
    from lambo.gui.pyplt.events import bind_quick_close

    plt.subplot(121)
    plt.hist(np.ravel(self.features))
    plt.title('features')

    plt.subplot(122)
    plt.hist(np.ravel(self.targets))
    plt.title('targets')

    bind_quick_close()
    plt.show()

  def test_window(self, size, win_num=1):
    batch = self.random_window(self, True, [size, size], win_num)
    batch.view()

  # endregion: Public Methods

  # region: APIs

  def rotation_test(
      self, model, angle_step=10, index=0, variation_diagram=False):
    # Get the designated interferogram and ground truth
    x, y = self.features[index], self.targets[index]
    features, targets = [], []

    # Fill in data
    extract = lambda im, a: Interferogram.get_downtown_area(
      Interferogram.rotate_image(im, a), p2=3)
    angles = list(range(0, 360, angle_step))
    for angle in angles:
      features.append(extract(x, angle))
      targets.append(extract(y, angle))

    # Get dataset
    features, targets = np.stack(features), np.stack(targets)
    dataset = PhaseSet(features, targets, name='Rotation-step-{}'.format(
      angle_step))

    # Evaluate model
    dataset.evaluate_model(model)

    # Show diagram if necessary
    if not variation_diagram: return

    import matplotlib.pyplot as plt
    plt.plot(angles, dataset[self.WMAE])
    plt.xlabel('Angle')
    plt.ylabel('Weighted MAE')
    plt.show()

  def evaluate_model(self, model):
    from tframe import Predictor

    assert isinstance(model, Predictor)
    console.show_status('Predicting {} ...'.format(self.name))

    # Predict
    y = model.predict(self, batch_size=1, verbose=True)
    self.data_dict[self.PREDICTED_KEY] = y
    self.data_dict[self.DELTA_KEY] = np.abs(y - self.targets)
    # Put details
    details, wmaes = [], []
    for truth, pred in zip(self.targets, y):
      wmae = self.wmae(truth, pred)
      detail = 'WMAE = {:.5f}'.format(wmae)
      detail += ', range(y) = [{:.3f}, {:.3f}]'.format(np.min(truth), np.max(truth))
      details.append(detail)
      wmaes.append(wmae)
    self.properties[self.EVAL_DETAILS] = details
    self.properties[self.WMAE] = wmaes
    self.view()

  def snapshot(self, model, index=0, over_trial=False, suffix='-final'):
    indices = [index]
    if over_trial: indices = [g[index] for g in self.groups.values()]
    for i in indices:
      self._snapshot(
        model, i, save_input=False, save_ground_truth=True,
        pred_suffix=suffix)

  def _snapshot(self, model, index=0, folder_path=None, save_input=False,
                save_ground_truth=True, pred_suffix=''):
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)
    y = model.predict(self[index], batch_size=1, verbose=False)
    y = np.reshape(y, y.shape[1:3])
    gt = self.targets[index].reshape(y.shape)
    x = self.features[index].reshape(y.shape)

    # Save input (if required), prediction and ground truth
    if folder_path is None: folder_path = model.agent.ckpt_dir
    suffix = '-{}-{}.png'.format(self.name, index)

    metric_str = '({:.4f})'.format(self.wmae(gt, y))
    for name, flag, img in zip(
        ('input', 'ground-truth', 'predicted' + metric_str + pred_suffix),
        (save_input, save_ground_truth, True), (x, gt, y)):
      if not flag: continue
      path = os.path.join(folder_path, name + suffix)
      if not os.path.exists(path): plt.imsave(path, img)

  @staticmethod
  def wmae(truth: np.ndarray, pred: np.ndarray):
    return np.sum(np.abs(truth - pred) * truth) / np.sum(truth)

  @staticmethod
  def random_window(batch: DataSet, is_training: bool,
                    shape: Union[tuple, list], num: int = 1,
                    rotate: bool = False):
    if not is_training: return batch
    h, w = shape

    full_shape = batch.features.shape[1:3]
    features, targets = [], []
    for x, y in zip(batch.features, batch.targets):
      # Rotate if required
      if rotate:
        angle = np.random.rand() * 360
        x, y = [Interferogram.get_downtown_area(Interferogram.rotate(im, angle))
                for im in (x, y)]
        full_shape = x.shape[0:2]

      # Choose the one among `num` regions which contains largest sample area
      ijs = [random_window(shape, full_shape) for _ in range(num)]

      # Sort ijs based on sample area
      if num > 1: ijs.sort(
        key=lambda ij: np.sum(y[ij[0]:ij[0]+h, ij[1]:ij[1]+w]), reverse=True)

      # Append selected window
      i, j = ijs[0]
      features.append(x[i:i+h, j:j+w])
      targets.append(y[i:i+h, j:j+w])

    # Set features and targets back
    batch.features = np.stack(features, axis=0)
    batch.targets = np.stack(targets, axis=0)

    return batch

  @staticmethod
  def random_window_preprocessor(
      shape: Union[tuple, list], num=1, random_rotate=False):
    return lambda batch, is_training: PhaseSet.random_window(
      batch, is_training, shape, num=num, rotate=random_rotate)

  @staticmethod
  def probe(trainer):
    from tframe.trainers.trainer import Trainer
    from .pr_configs import PRConfig

    # Sanity check
    th = trainer.th
    assert isinstance(trainer, Trainer) and isinstance(th, PRConfig)

    # Get indices from th
    train_indices, val_indices, test_indices = th.probe_indices
    if not train_indices and not test_indices: return

    # Take snap short for specified samples
    for data_set, indices in zip(
        (trainer.training_set, trainer.validation_set, trainer.test_set),
        (train_indices, val_indices, test_indices)):
      for i in indices:
        assert isinstance(data_set, PhaseSet)
        for g in data_set.groups.values(): data_set._snapshot(
          trainer.model, g[i], save_ground_truth=True,
          pred_suffix='-{:.1f}'.format(trainer.total_rounds))

    return 'Snapshot saved to checkpoint folder.'

  # endregion: APIs

