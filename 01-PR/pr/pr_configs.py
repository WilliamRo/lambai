from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class PRConfig(SmartTrainerHub):

  class PRKeys(object):
    prior = 'PR_PRIOR'

  feature_type = Flag.integer(
    1, '1 for interferogram, 2 for 2-D extracted image', is_key=None)
  radius = Flag.integer(None, '$k_0 \cdot NA$', is_key=None)

  fn_pattern = Flag.string(None, 'Pattern filter for data folders', is_key=None)

  train_indices = Flag.string(
    None, 'Sample indices for training set', is_key=None)
  val_indices = Flag.string(
    None, 'Sample indices for validation set', is_key=None)
  test_indices = Flag.string(
    None, 'Sample indices for test set', is_key=None)

  pr_dev_code = Flag.string('-', 'Developer code', is_key=None)

  mask_min = Flag.float(0.1, 'Minimum value for loss mask', is_key=None)
  mask_alpha = Flag.float(
    0.1, 'Parameter alpha used in generating soft mask', is_key=None)

  win_size = Flag.integer(None, 'Random window size', is_key=None)

  # Encoder-Decoder related
  half_height = Flag.integer(
    None, 'Times of contracting/expanding', is_key=None)
  thickness = Flag.integer(
    None, 'Additional layers used Encoder-Decoder structure', is_key=None)
  use_maxpool = Flag.boolean(
    None, 'Whether to use maxpool for contracting', is_key=None)
  bridges = Flag.string(None, 'Link indices used in UNet2D', is_key=None)

  # Data
  truncate_at = Flag.float(12, 'Max value of targets')

  win_num = Flag.integer(
    1, 'Number of windows in PhaseSet.random_windows', is_key=None)
  random_rotate = Flag.boolean(
    False, 'Whether to rotate training images randomly', is_key=None)

  eval_rotation = Flag.boolean(False, 'Whether to evaluate rotation')


  # Probe related
  train_probe_ids = Flag.string(None, 'Sample indices in train set for probing')
  val_probe_ids = Flag.string(None, 'Sample indices in val set for probing')
  test_probe_ids = Flag.string(None, 'Sample indices in test set for probing')

  epoch_per_probe = Flag.integer(2, 'Epoch per probe')

  # TODO: BETA
  use_prior = Flag.boolean(False, 'Whether to use prior', is_key=None)
  prior_size = Flag.integer(None, 'Size of prior map', is_key=None)
  hyper_filter_size = Flag.integer(3, 'Hyper filter size', is_key=None)
  prior_key = Flag.string('cube', r'\in (`cube`, `dettol`)', is_key=None)
  prior_format = Flag.string('real', r'\in (`real`, `complex`)', is_key=None)

  kon_activation = Flag.string(
    None, 'activation function used in konjac', is_key=None)

  kon_omega = Flag.integer(30, '...', is_key=None)
  kon_rs = Flag.list(None, '...')


  def train_val_test_indices(self):
    def _parse(indices_string: str) -> tuple:
      indices = [int(i) for i in indices_string.split(',')]
      assert len(indices) == len(set(indices))
      for i in indices: assert i > 0
      return tuple(indices)
    return [_parse(ind) for ind in (
      self.train_indices, self.val_indices, self.test_indices)]

  @property
  def probe_indices(self) -> [list, list, list]:
    train_indices, val_indices, test_indices = [], [], []

    for id_string, indices in zip(
        (self.train_probe_ids, self.val_probe_ids, self.test_probe_ids),
        (train_indices, val_indices, test_indices)):
      if id_string in (None, 'x', '-'): continue
      assert isinstance(id_string, str)
      indices.extend([int(s) for s in id_string.split(',')])
      assert len(indices) == len(set(indices))

    return train_indices, val_indices, test_indices


# New hub class inherited from SmartTrainerHub must be registered
PRConfig.register()