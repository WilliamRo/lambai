from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class PRConfig(SmartTrainerHub):

  radius = Flag.integer(None, '$k_0 \cdot NA$', is_key=None)

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


  def train_val_test_indices(self):
    def _parse(indices_string: str) -> tuple:
      indices = [int(i) for i in indices_string.split(',')]
      assert len(indices) == len(set(indices))
      for i in indices: assert i > 0
      return tuple(indices)
    return [_parse(ind) for ind in (
      self.train_indices, self.val_indices, self.test_indices)]


# New hub class inherited from SmartTrainerHub must be registered
PRConfig.register()