import pr_core as core
import pr_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'acorn'
id = 0
def model():
  th = core.th
  model = m.get_container()

  def add_conv(filters):
    return model.add(m.m.Conv2D(
      filters, th.kernel_size, use_bias=False,
      use_batchnorm=th.use_batchnorm, activation=th.activation))

  for n in th.archi_string.split('-'):
    add_conv(int(n))

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

  th.fn_pattern = '0[45]-'

  indices = '1,2'
  th.train_indices = indices
  th.val_indices = indices
  th.test_indices = indices

  th.train_config = 't10'
  th.val_config = th.train_config

  th.test_config = '-10t'

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.random_rotate = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.7
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.kernel_size = 9
  th.activation = 'relu'
  th.archi_string = '32-32-24-16-8-4'

  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.loss_string = 'wmae:0.0001'
  # th.loss_string = 'mse'

  th.epoch = 50000
  th.batch_size = 16
  th.updates_per_round = 30
  th.validation_per_round = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  th.patience = 40
  th.early_stop = True
  th.save_model = True

  # Commander Center
  th.eval_rotation = False
  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.print_cycle = 1
  th.train_probe_ids = '0'
  # th.val_probe_ids = '0'
  th.test_probe_ids = '0'
  th.epoch_per_probe = 10

  th.suffix = '_tl' + ''.join(th.train_indices.split(','))
  th.suffix += '(' + th.train_config + ')'
  th.suffix += '_' + th.loss_string.replace(':', '')
  if th.feature_type != 1: th.suffix += '_ft{}'.format(th.feature_type)
  if th.random_rotate: th.suffix += '_rotate'

  tail = '(k{}-{}-{})'.format(th.kernel_size, th.archi_string, th.activation)
  th.mark = '{}{}'.format(model_name, tail)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  # For test
  if False:
    th.feature_type = 2
    th.prefix = '0511_'
    th.train = False
    th.eval_rotation = True
    # th.eval_rotation = False

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

