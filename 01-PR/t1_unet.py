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

  m.m.UNet2D(
    th.filters, th.kernel_size, th.activation, th.half_height, th.thickness,
    use_maxpool=th.use_maxpool, use_batchnorm=th.use_batchnorm,
    link_indices=th.bridges).add_to(model)

  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.win_size = 512
  th.win_num = 1

  # th.train_indices = '1,2,3'
  # th.val_indices = '1,2,3'
  # th.test_indices = '4'

  th.train_indices = '3,4'
  th.val_indices = '3,4'
  th.test_indices = '3,4'

  th.train_config = ':1'
  th.val_config = '51:52'
  th.test_config = '51:52'

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_onepiece'

  # th.pr_dev_code = 'dev0-2'
  # th.aug_config = 'rotate|flip'

  th.allow_growth = False
  th.gpu_memory_fraction = 0.7
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  # | 1024 | 512 | 256 | 128 | 64 | 32 | 16 |
  # |   0  |  1  |  2  |  3  |  4 |  5 |  6 |
  th.filters = 16
  th.kernel_size = 3
  # th.activation = 'lrelu:0.2'   # [YY]
  th.activation = 'relu'
  th.half_height = 4
  th.thickness = 2
  th.bridges = 'a'

  th.use_batchnorm = True
  # th.bn_momentum = 0.8    # [YY]
  th.use_maxpool = True
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

  # th.global_l2_penalty = 0.0001

  th.patience = 40
  th.early_stop = True
  th.save_model = True

  # Commander Center
  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.print_cycle = 1
  th.train_probe_ids = '0'
  th.test_probe_ids = '0'
  th.epoch_per_probe = 5

  tail = ''
  th.mark = '{}{}'.format(model_name, tail)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  # For test
  if False:
    th.prefix = '0412_'
    th.suffix = '_2'
    th.mark = 'acorn'
    th.train = False

    th.train_indices = '3,4'
    th.val_indices = '3,4'
    th.test_indices = '1,2'

    th.train_config = '-'
    th.val_config = '-'
    th.test_config = '-'

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

