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
  PRIOR = 'prior'

  # region: Properties

  @property
  def prior(self) -> np.ndarray:
    return self[self.PRIOR]

  @prior.setter
  def prior(self, val: np.ndarray):
    assert len(val) == len(self.features)
    self.data_dict[self.PRIOR]  = val

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

  def set_prior(self, angle=0):
    from pr_core import th
    assert th.use_prior
    if th.prior_key == 'cube':
      self.prior = np.stack([
        ig.get_fourier_prior(th.prior_size, angle=angle)
        for ig in self.interferograms])
    elif th.prior_key == 'dettol':
      self.prior = np.stack([
        ig.get_fourier_prior_stack(
          th.prior_size, angle=angle, omega=th.kon_omega, rs=th.kon_rs,
          fmt=th.prior_format)
        for ig in self.interferograms])
    elif th.prior_key == 'dual':
      # shape = [?, 2, 2], prior[k, 0] is center, prior[k, 1] is unit vector
      self.prior = np.stack([ig.get_fourier_dual_basis(peak_and_uv=True)
                             for ig in self.interferograms])

      # TODO: to be cleared
      # self.prior = np.real(np.stack([
      #   np.stack(ig.get_fourier_dual_basis(
      #     th.prior_size, omega=th.kon_omega,
      #     rs=th.kon_rs, rotundity=th.rotundity), axis=-1)
      #   for ig in self.interferograms]))
    else: raise KeyError

  def reset_feature(self, feature_type):
    assert feature_type in (2, 3, 9)
    console.show_status('Resetting feature type ...')
    features = np.stack(
      [ig.get_model_input(feature_type) for ig in self.interferograms], axis=0)
    self.features = features
    console.show_status('Feature reset to type {}'.format(feature_type))

  def add_channel(self):
    if len(self.features.shape) == 3:
      self.features = np.expand_dims(self.features, axis=-1)
    assert len(self.features.shape) == 4

    if len(self.targets.shape) == 3:
      self.targets = np.expand_dims(self.targets, axis=-1)
    assert len(self.targets.shape) == 4

  def normalize_features(self):
    x = self.features
    assert len(x.shape) == 4

    shape = [len(x), 1, 1, x.shape[3]]
    mu = np.mean(x, axis=(1, 2)).reshape(shape)
    sigma = np.std(x, axis=(1, 2)).reshape(shape)
    # Set back features
    self.features = (x - mu) / sigma

  def squash_target(self, truncate_at=12.0):
    assert truncate_at > 0
    assert np.min(self.targets) == 0
    self.targets[self.targets > truncate_at] = truncate_at
    self.targets = self.targets / truncate_at

  def view(self):
    from lambo import DaVinci

    ds = self[:]
    # Set features accordingly
    x = self.features
    assert len(x.shape) == 3 or len(x.shape) == 4 and x.shape[-1] in (1, 2)
    if len(x.shape) == 3 or x.shape[-1] == 1:
      horizontal_list = ['targets', 'features']
    else:
      horizontal_list = ['targets', 'features[0]', 'features[1]']
      ds.data_dict['features[0]'] = x[:, :, :, 0]
      ds.data_dict['features[1]'] = x[:, :, :, 1]

    # Fill horizontal_keys accordingly
    for key in (self.DELTA_KEY, self.PREDICTED_KEY, self.PRIOR):
      if key in self.data_dict: horizontal_list.append(key)

    # Visualize using DaVinci
    da = DaVinci(self.name)
    da._color_bar = True
    da.objects = [self[i] for i in range(self.size)]

    def get_plotter(key: str = key):
      def _plotter(x: DataSet):
        im: np.ndarray = x[key]
        if im.shape[0] == 1: im = im[0]
        if im.shape[-1] == 1: im = im[:, :, 0]

        # Set color limit according
        if key == 'features': clim = [None, None]
        else: clim = [f(x['targets']) for f in (np.min, np.max)]
        da._color_limits = clim

        da.imshow_pro(im)
        da.title = key + f' - {self.name}'
      return _plotter

    # Add plotters
    for key in horizontal_list: da.add_plotter(get_plotter(key=key))

    da.show()

  def view_aberration(self):
    from lambo.gui.vinci.vinci import DaVinci

    da = DaVinci('Aberration Viewer')
    da.keep_3D_view_angle = True
    da.z_lim_tuple = (-50, 30)
    da.objects = self.interferograms

    def _plot_aberration(x, ax, bg=False, plot3d=False):
      assert isinstance(x, Interferogram)
      ig = x._backgrounds[0] if bg else x
      im = ig.extracted_angle_unwrapped
      # im = ig.extracted_angle
      title = 'Background' if bg else 'Sample'
      title += ', peak at {}'.format(ig.peak_index)
      # Plot accordingly
      if plot3d: da.plot3d(im, ax, title=title)
      else: da.imshow(im, ax, color_bar=True, title=title)

    da.add_plotter(_plot_aberration)
    da.add_plotter(lambda x, ax: _plot_aberration(x, ax, True))
    da.add_plotter(lambda x, ax3d: _plot_aberration(x, ax3d, plot3d=True))
    da.add_plotter(lambda x, ax3d: _plot_aberration(x, ax3d, True, plot3d=True))
    da.show()

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
    r = re.match(r'^-?(?:0\.\d*)?\d*([at])-?(?:0\.\d*)?\d*$', config_str)
    assert r is not None

    # Define some utilities
    def parse_index(s: str, total: int, is_begin: bool):
      if s == '': return 0 if is_begin else total
      n = float(s)
      sign = np.sign(n)
      n *= sign
      if n < 1: n = int(total * n)
      assert int(n) == n
      return int(sign * n)

    # Get begin and end
    sep = r.group(1)
    begin, end = config_str.split(sep)

    # Get subset
    if sep == 't':
      # Case (1) get samples from each trial
      indices = []
      for g in self.groups.values():
        _begin = parse_index(begin, len(g), is_begin=True)
        _end = parse_index(end, len(g), is_begin=False)
        indices.extend(g[_begin:_end])
      subset = self[indices]
    else:
      assert sep == 'a'
      # Case (2) otherwise get samples for whole set
      begin = parse_index(begin, self.size, is_begin=True)
      end = parse_index(end, self.size, is_begin=False)
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
      self, model, angle_step=10, index=0, variation_diagram=False,
      data_leak=False):
    from pr_core import th

    # Get the designated interferogram and ground truth
    x, y = self.features[index], self.targets[index]
    features, targets, priors = [], [], []

    # Fill in data
    extract = lambda im, a, p2=3: Interferogram.get_downtown_area(
      Interferogram.rotate_image(im, a), p2=p2)
    angles = list(range(0, 360, angle_step))
    for angle in angles:
      features.append(extract(x, angle))
      targets.append(extract(y, angle))

      # Rotate prior if necessary
      if th.use_prior:
        ig = self.interferograms[index]
        if th.prior_key == 'dettol':
          priors.append(ig.get_fourier_prior_stack(
            th.prior_size, angle, th.kon_omega, th.kon_rs))
        elif th.prior_key == 'dual':
          priors.append(np.real(np.stack(ig.get_fourier_dual_basis(
            th.prior_size, th.kon_omega, th.kon_rs,
            angle=angle, rotundity=th.rotundity), axis=-1)))
        else: priors.append(ig.get_fourier_prior(th.prior_size, angle))

    # Get dataset
    features, targets = np.stack(features), np.stack(targets)
    dataset = PhaseSet(features, targets, name='Rotation-step-{}'.format(
      angle_step))

    # Set prior if necessary
    if th.use_prior: dataset.prior = np.stack(priors)

    if data_leak: return dataset
    # Evaluate model
    dataset.evaluate_model(model)

    # Show diagram if necessary
    if not variation_diagram: return

    import matplotlib.pyplot as plt
    plt.plot(angles, dataset[self.WMAE])
    plt.xlabel('Angle')
    plt.ylabel('Weighted MAE')
    plt.show()

  def evaluate_model(self, model, view=True):
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
    if view: self.view()
    return y

  def snapshot(self, model, index=0, over_trial=False, step='final-'):
    indices = [index]
    if over_trial: indices = [g[index] for g in self.groups.values()]
    for i in indices: self._snapshot(
      model, i, save_input=False, save_ground_truth=True, step=step)

  def _snapshot(self, model, index=0, folder_path=None, save_input=False,
                save_ground_truth=True, step=''):
    from tframe import Predictor
    from pr_core import th
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)
    y = model.predict(self[index], batch_size=1, verbose=False)
    y = np.reshape(y, y.shape[1:3])
    gt = self.targets[index].reshape(y.shape)
    if th.feature_type == 1:
      x = self.features[index].reshape(y.shape)
    else:
      x = self.features[index][:, :, 0].reshape(y.shape)

    # Save input (if required), prediction and ground truth
    if folder_path is None: folder_path = model.agent.ckpt_dir
    suffix = '-{}-{}.png'.format(self.name, index)

    # Get metrics
    val_dict = model.validate_model(self[index], allow_sum=False)

    metric_str = '-'.join([f'{k.name}{v:.4f}' for k, v in val_dict.items()])
    for name, flag, img in zip(
        ('input', 'ground-truth', step + metric_str),
        (save_input, save_ground_truth, True), (x, gt, y)):
      if not flag: continue
      path = os.path.join(folder_path, name + suffix)
      if not os.path.exists(path): plt.imsave(path, img)

  def dump_package(self, model):
    from tframe import Predictor
    from tframe import context
    from pr_core import th
    import os,  pickle

    if not th.use_dual_conv: return
    assert isinstance(model, Predictor)

    path = os.path.join(model.agent.ckpt_dir, 'misc.dict')
    with open(path, 'wb') as f:
      pickle.dump(context.note.misc, f, pickle.HIGHEST_PROTOCOL)
    console.show_status(f'note.misc dumped to `{path}`')

  @staticmethod
  def wmae(truth: np.ndarray, pred: np.ndarray):
    return np.sum(np.abs(truth - pred) * truth) / np.sum(truth)

  @staticmethod
  def random_window(batch: DataSet, is_training: bool,
                    shape: Union[tuple, list], num: int = 1,
                    rotate: bool = False):
    assert isinstance(batch, PhaseSet)
    from pr_core import th

    if not is_training:
      # For validation and test set
      if th.use_prior:
        batch.prior = batch.prior[:, :th.prior_size, :th.prior_size]
      return batch
    h, w = shape

    full_shape = batch.features.shape[1:3]
    features, targets, priors = [], [], []
    for i, (x, y) in enumerate(zip(batch.features, batch.targets)):
      # Rotate if required
      p = batch.prior[i] if th.use_prior else None
      if rotate:
        angle = np.random.rand() * 360
        x, y = [Interferogram.get_downtown_area(
          Interferogram.rotate_image(im, angle)) for im in (x, y)]
        full_shape = x.shape[0:2]

        # Rotate prior if used (at this time prior is not cropped yet)
        if p is not None:
          assert not th.use_dual_conv  # TODO:
          # Retrieve the interferogram
          ig = batch.interferograms[i]
          p = ig.get_fourier_prior(th.prior_size, angle)

      # Choose the one among `num` regions which contains largest sample area
      ijs = [random_window(shape, full_shape) for _ in range(num)]

      # Sort ijs based on sample area
      if num > 1: ijs.sort(
        key=lambda ij: np.sum(y[ij[0]:ij[0]+h, ij[1]:ij[1]+w]), reverse=True)

      # Append selected window
      i, j = ijs[0]
      features.append(x[i:i+h, j:j+w])
      targets.append(y[i:i+h, j:j+w])

      # Set prior if necessary
      if th.use_prior: priors.append(p[:th.prior_size, :th.prior_size])

    # Set features and targets back
    batch.features = np.stack(features, axis=0)
    batch.targets = np.stack(targets, axis=0)

    # Set prior if necessary
    if th.use_prior: batch.prior = np.stack(priors, axis=0)

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
          step='{:06.1f}-'.format(trainer.total_rounds))

    message = 'Snapshot saved to checkpoint folder'
    if not th.use_dual_conv: return message

    # Pupil logic:

    # Take down theta, r, and radius
    from tframe import context

    STEP, PACKAGE = 'STEP', 'PACKAGE'
    # Initialize if necessary
    if STEP not in context.note.misc:
      for key in (STEP, PACKAGE): context.note.misc[key] = []

    # Record step
    context.note.misc[STEP].append(trainer.total_rounds)
    package = trainer.session.run(context.get_collection_by_key('dual'))
    context.note.misc[PACKAGE].append(package)

    _, d, r = package
    console.show_info(f'd: [{np.min(d):.3f}, {np.max(d):.3f}], '
                      f'r: [{np.min(r):.3f}, {np.max(r):.3f}]')

    return message + '. Pupil shape exported.'

  # endregion: APIs

  # region: Model Visualization


  # endregion: Model Visualization

