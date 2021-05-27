import pr_core as core
import pr_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'unet'
id = 5
def model():
  th = core.th

  bridges = th.bridges
  if bridges not in (None, '-', 'a', 'x'):
    bridges = [int(s) for s in bridges.split(',')]

  unet = m.m.UNet2D(
    th.filters, activation=th.activation, height=th.half_height,
    thickness=th.thickness, use_maxpool=th.use_maxpool,
    use_batchnorm=th.use_batchnorm,
    contraction_kernel_size=th.contraction_kernel_size,
    expansion_kernel_size=th.expansion_kernel_size,
    link_indices=bridges)

  th.mark += '({})'.format(str(unet))

  model = m.get_container()
  unet.add_to(model)

  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.feature_type = 1
  th.win_size = 512
  th.win_num = 1

  th.fn_pattern = '05-'
  th.fn_pattern = '0[45]-'

  indices = '1'
  indices = '1,2'
  th.train_indices = indices
  th.val_indices = indices
  th.test_indices = indices

  th.int_para_1 = -10
  th.train_config = 'a{}'.format(th.int_para_1)
  th.val_config = '{}a'.format(th.int_para_1)

  th.train_config = 't10'
  th.val_config = th.train_config

  th.test_config = '-10t'
  # th.val_config = th.test_config

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.random_rotate = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.75
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.filters = 16
  th.contraction_kernel_size = 9
  th.expansion_kernel_size = 3
  th.activation = 'relu'
  th.half_height = 3
  th.thickness = 2
  th.bridges = 'a'

  th.use_batchnorm = False
  # th.bn_momentum = 0.8    # [YY]
  th.use_maxpool = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.loss_string = 'wmae:0.0001'
  # th.loss_string = 'mae'

  th.epoch = 50000
  th.batch_size = 16
  th.updates_per_round = 30
  th.validation_per_round = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  # th.global_l2_penalty = 0.0001

  th.patience = 20
  th.early_stop = True
  th.save_model = True

  # Commander Center
  th.eval_rotation = False
  th.train = True
  th.overwrite = False
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.print_cycle = 1
  th.train_probe_ids = '0'
  th.val_probe_ids = '0'
  # th.test_probe_ids = '0'

  th.epoch_per_probe = 10

  th.suffix = '_tl' + ''.join(th.train_indices.split(','))
  th.suffix += '(' + th.train_config + ')'
  th.suffix += '_' + th.loss_string.replace(':', '')
  if th.feature_type != 1: th.suffix += '_ft{}'.format(th.feature_type)
  if th.random_rotate: th.suffix += '_rotate'

  tail = ''
  th.mark = '{}{}'.format(model_name, tail)
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  # For test
  if True:
    # th.feature_type = 1
    th.prefix = '0515_'
    th.train = False
    th.feature_type = 1
    # th.visualize_tensors = True
    th.eval_rotation = False

  # th.allow_activation = False
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

