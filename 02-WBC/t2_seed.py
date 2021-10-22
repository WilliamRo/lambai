import wbc_core as core
import wbc_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.organizer.task_tools import update_job_dir
from tframe.utils.misc import date_string

from tframe.nets.classic.conv_nets.lenet import LeNet


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'seed'
id = 0
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False, add_last_dim=True)

  # Parse
  conv_list, fc_list = LeNet.parse_archi_string(th.archi_string)

  # Add conv layers
  for i, filters in enumerate(conv_list):
    model.add(m.Conv2D(filters, th.kernel_size, th.strides))
    if i > 0 and th.use_batchnorm:
      model.add(m.BatchNormalization())
    model.add(m.Activation(th.activation))
  # Add flatten layer
  model.add(m.Flatten())
  # Add fully-connected layers
  m.add_rethink(th, model, fc_list, index_group=(0, 1))

  # Add re-thinker structure
  return m.finalize(th, model, add_output_layer=False, flatten=False)


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.image_side_length = 300

  th.centralize_data = True

  th.data_config = 'x'
  th.val_config = 'c-!r-100'
  th.test_config = 'c-!r-100'

  th.augmentation = True
  th.aug_config = 'flip|rotate'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  th.gpu_memory_fraction = 0.37
  th.allow_growth = False
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.archi_string = '12-16-24-32=32-16'
  th.archi_string = m.get_archi_string()
  th.kernel_size = 5
  th.strides = 3
  th.activation = 'relu'

  th.use_batchnorm = False

  th.use_wise_man = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 32  # TODO: try less than 32

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0003   # TODO: not stable

  th.patience = 5         # TODO: try larger than 10
  th.early_stop_metric = 'f1'

  th.train = True
  # ---------------------------------------------------------------------------
  # 4. display, summary and note setup
  # ---------------------------------------------------------------------------
  th.validate_test_set = True
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})_bs({})_val({})_tes({})'.format(
    model_name, th.archi_string, th.batch_size, th.val_config, th.test_config)
  core.activate(True)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


"""
Best HPs:
batch_size: 32
lr: 0.0003 (Adam)
patience: 10
use_wise_man: True
archi_string: 12-16-24-32=32-16
augmentation: 'flip|rotate'
activation: relu
centralize_data: True
kernel_size: 5
pad_mode: constant
strides: 3

"""

