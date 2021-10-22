import wbc_core as core
import wbc_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.classic.conv_nets.lenet import LeNet


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'lenet'
id = 2
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False, add_last_dim=True)
  LeNet(archi_string=th.archi_string, kernel_size=th.kernel_size,
        strides=th.strides, activation=th.activation,
        padding=th.padding).add_to(model)
  return m.finalize(th, model)


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.image_height = 300
  th.image_width = 300

  th.centralize_data = True
  th.test_config = 'd-3'
  th.val_config = 'd-2'
  th.test_config = 'c-!r-100'
  th.val_config = 'c-!r-100'

  th.augmentation = True
  th.aug_config = 'flip|rotate'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_t00'
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  # th.archi_string = '6-16-24=120-84'
  th.archi_string = '12-16-24-32=32-16'
  th.kernel_size = 5
  th.strides = 3
  th.activation = 'relu'
  th.padding = 'same'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 64
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0003

  th.patience = 7
  th.early_stop = True
  th.early_stop_metric = 'f1'
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True

  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})_{}_s{}_k{}'.format(
    model_name, th.archi_string, th.activation, th.strides, th.kernel_size)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

