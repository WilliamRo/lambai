import cf10_core as core
import cf10_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir
from tframe.nets.classic.conv_nets.lenet import LeNet


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'lenet'
id = 1
def model():
  th = core.th
  model = m.get_container(flatten=False)
  LeNet(archi_string=th.archi_string, kernel_size=th.kernel_size,
        strides=th.strides, activation=th.activation).add_to(model)
  return m.finalize(model)


def main(_):
  console.start('{} on CIFAR-10 task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.centralize_data = True

  th.augmentation = True
  th.aug_config = 'flip:True;False'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_t00'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '64-32=120-84'
  th.kernel_size = 4
  th.strides = 2
  th.activation = 'relu'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 64
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0003

  th.patience = 5
  th.early_stop = True
  th.save_model = True

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})_{}_s{}_k{}'.format(
    model_name, th.archi_string, th.activation, th.strides, th.kernel_size)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate(True)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

