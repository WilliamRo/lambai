from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console


class PRConfig(SmartTrainerHub):

  class PRKeys(object):
    prior = 'PR_PRIOR'

  fix_input_size = Flag.boolean(
    False, 'Whether to fix the size of input image')

  feature_type = Flag.integer(
    1, '1 for interferogram, 2 for 2-D extracted image', is_key=None)
  radius = Flag.integer(None, '$k_0 \cdot NA$', is_key=None)

  fn_pattern = Flag.string(None, 'Pattern filter for data folders', is_key=None)
  re_pattern = Flag.string(None, 'RE Pattern filter for data folders',
                           is_key=None)

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

  edge_cut = Flag.integer(0, 'Dataset edge cut', is_key=None)

  # Encoder-Decoder related
  half_height = Flag.integer(
    None, 'Times of contracting/expanding', is_key=None)
  thickness = Flag.integer(
    None, 'Additional layers used Encoder-Decoder structure', is_key=None)
  use_maxpool = Flag.boolean(
    None, 'Whether to use maxpool for contracting', is_key=None)
  bridges = Flag.string(None, 'Link indices used in UNet2D', is_key=None)
  guest_first = Flag.boolean(False, 'Used in bridge')

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
  rotundity = Flag.boolean(False, '...', is_key=None)

  kon_activation = Flag.string(
    None, 'activation function used in konjac', is_key=None)

  kon_omega = Flag.integer(30, '...', is_key=None)
  kon_rs = Flag.list(None, '...')
  kon_rs_str = Flag.string(None, '...', is_key=None)
  kon_rad = Flag.float(1.0, 'Pupil radius percentage', is_key=None)

  use_dual_conv = Flag.boolean(
    False, 'Whether to use dual convolution', is_key=None)

  hyper_dual_num = Flag.integer(0, 'Currently should be 0', is_key=None)

  n2o = Flag.float(1.0, 'Pupil booster', is_key=None)

  local_activation = Flag.string('-', 'Naphtali local activation', is_key=None)
  global_activation = Flag.string(
    '-', 'Naphtali global activation', is_key=None)
  nap_merge = Flag.string('concat', 'Naphtali merge method', is_key=None)
  nap_token = Flag.string('alpha', 'Naphtali block type', is_key=None)
  ash_token = Flag.string('alpha', 'Ashaer block type', is_key=None)

  data_token = Flag.string(None, 'Data token', is_key=None)
  group_indices = Flag.whatever([], '...')

  # final_activation = Flag.


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

  def data_setup(self, token='alpha'):
    from pr_core import th
    if token == 'alpha':
      th.fn_pattern = '0[45]-'
      indices = '1,2'
      th.train_indices = indices
      th.val_indices = indices
      th.test_indices = indices

      th.train_config = 't5'
      th.val_config = self.train_config
      th.test_config = '-5t'

      th.train_probe_ids = '0'
      th.test_probe_ids = '0'
      th.probe_cycle = 10
    elif token == 'beta':
      # 05-beads #3: single image of 6 beads
      th.fn_pattern = '05-'
      th.train_indices = th.val_indices = th.test_indices = '1'
      th.train_config = th.val_config = '2a3'
      th.test_config = 'a'

      th.train_probe_ids = '0'
      th.test_probe_ids = '4'
    elif token == 'gamma':
      th.fn_pattern = '05-'
      th.train_indices = th.val_indices = th.test_indices = '1'
      th.train_config = th.val_config = '4a5'
      th.test_config = 'a'

      th.train_probe_ids = '0'
      th.test_probe_ids = '2'
    elif token == 'delta':
      th.fn_pattern = '04-'
      th.train_indices = th.val_indices = th.test_indices = '1'
      th.train_config = th.val_config = 'a1'
      th.test_config = 'a2'

      th.train_probe_ids = '0'
      th.test_probe_ids = '1'
      # th.win_num = 4
    elif token == 'epsilon':
      th.fn_pattern = '03-'
      th.train_indices = th.val_indices = th.test_indices = '1'
      th.val_config = th.train_config
      th.test_config = 'a'

      th.test_probe_ids = '0,1,2,3,4,5,7'
      th.group_indices = [0, 1, 2, 3, 4, 5, 7]
    elif token == 'zeta':
      # Train on RBC 001 - sparse
      th.fn_pattern = '04-'
      th.train_indices = th.val_indices = th.test_indices = '1'
      th.train_config = th.val_config = 'a1'
      th.test_config = 'a'
      th.train_probe_ids = '0'
      th.test_probe_ids = '6'
    elif token == 'eta':
      # Train on dense bead - sparse
      th.fn_pattern = '05-'
      th.train_indices = th.val_indices = th.test_indices = '1'
      th.val_config = th.train_config
      th.test_config = 'a'

      th.test_probe_ids = '2,4,39'
      th.group_indices = [2, 4, 39]
      if 'jan29' in th.developer_code:
        th.test_probe_ids = '1,3,14'
        th.group_indices = [1, 3, 14]

      # th.edge_cut = 8
    else: raise NotImplementedError
    console.show_status(f'Applied data setup {token}.', 'yellow')

  def trainer_setup(self, token='alpha'):
    from pr_core import th
    if token == 'alpha':
      th.loss_string = 'xmae' if 0 <= th.alpha < 1 else 'wmae:0.0001'

      th.epoch = 50000
      th.updates_per_round = 30
      th.batch_size = 16
      th.validation_per_round = 1

      th.optimizer = 'adam'
      th.learning_rate = 0.0001

      th.patience = 30
      th.early_stop = True
      th.save_model = True
      th.epoch_per_probe = 10

      #
      th.print_cycle = 1
      th.train_probe_ids = '0'
      th.test_probe_ids = '0'
    elif token == 'beta':
      th.epoch = 50000
      th.updates_per_round = 30
      th.batch_size = 16
      th.validation_per_round = 1

      th.optimizer = 'adam'
      th.learning_rate = 0.0001

      th.patience = 5
      th.epoch_per_probe = 5

      th.print_cycle = 5
    elif token == 'gamma':
      th.epoch = 400
      th.early_stop = False
      th.epoch_per_probe = 10
    else: raise NotImplementedError
    console.show_status(f'Applied trainer setup {token}.', 'yellow')

  def unet_setup(self, f=8, cks=3, eks=3, act='relu', bn=True,
                 bri='a', h=4, thk=2, duc=False, mp=False):
    from pr_core import th
    th.filters = f
    th.contraction_kernel_size = cks
    th.expansion_kernel_size = eks
    th.activation = act
    th.bottle_neck = bn
    th.bridges = bri
    th.half_height = h
    th.thickness = thk
    th.use_duc = duc
    th.use_maxpool = mp
    console.show_status(f'UNet has been setup', 'yellow')

  def dual_setup(self):
    from pr_core import th
    if not th.use_dual_conv: return
    th.use_prior = True
    th.prior_size = 25
    th.prior_key = 'dual'
    th.kon_rs_str = '0.5'
    th.kon_rs = [float(s) for s in th.kon_rs_str.split(',')]
    th.kon_omega = 10
    th.kon_rad = 0.9
    th.n2o = 10.0
    console.show_status(f'DualConv has been setup', 'yellow')

  def dual_suffix(self):
    if self.hyper_dual_num > 0:
      return f'(hd-{self.hyper_dual_num})'
    else: return '({})'.format('-'.join(
      [f'L-{self.prior_size}',
        f'omega-{self.kon_omega}',
       'rs-{}'.format('-'.join([str(r) for r in self.kon_rs]))]))


# New hub class inherited from SmartTrainerHub must be registered
PRConfig.register()