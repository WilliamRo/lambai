import wbc_core as core
import wbc_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'resnet'
id = 1
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False, add_last_dim=True)

  # First block
  filters = th.int_para_1
  model.add(m.Conv2D(filters, kernel_size=3, strides=2))
  m.add_bn_relu(model)
  model.add(m.MaxPool2D(2, strides=2))

  # Parse architecture_string
  repetitions = [int(c) for c in th.archi_string.split('-')]

  # Add residual blocks
  for block_id, r in enumerate(repetitions):
    m.add_residual_unit(model, filters, repetitions=r, unit_id=block_id)
    filters *= 2

  # Last building block
  m.add_bn_relu(model)
  model.add(m.GlobalAveragePooling2D())
  return m.finalize(th, model, flatten=True)


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.image_height = 300
  th.image_width = 300

  th.centralize_data = True
  th.augmentation = True
  th.aug_config = 'flip|rotate'

  th.data_config = 'x'
  th.val_config = 'd-2'
  th.test_config = 'd-3'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_t00'
  th.visible_gpu_id = 0

  th.gpu_memory_fraction = 0.8
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.archi_string = '1-2-1'
  th.int_para_1 = 32  # Initial filters

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 32
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.003

  th.patience = 6
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
  th.mark = '{}({})_bs{}'.format(model_name, th.archi_string, th.batch_size)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  # Use this line to avoid memory allocation error
  th.val_batch_size = th.batch_size
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

