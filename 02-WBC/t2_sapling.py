import wbc_core as core
import wbc_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'sapling'
id = 9
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False, add_last_dim=True)

  # Build classic AlexNet
  model.add(m.Conv2D(8, kernel_size=3, strides=1, activation='relu'))
  model.add(m.MaxPool2D(2, strides=2))

  model.add(m.Conv2D(16, kernel_size=3, strides=1, activation='relu'))
  model.add(m.MaxPool2D(2, strides=2))

  for filters in (24, 18):
    model.add(m.Conv2D(filters, kernel_size=3))
    model.add(m.Activation('relu'))

  model.add(m.MaxPool2D(3, strides=2))
  model.add(m.Flatten())

  for dim in (64, 64):
    if th.dropout > 0: model.add(m.Dropout(1. - th.dropout))
    model.add(m.Dense(dim, activation='relu'))

  return m.finalize(th, model)


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.image_height = 300
  th.image_width = 300

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_t00'

  th.visible_gpu_id = 0
  th.allow_growth = True
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.dropout = 0.0
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 32
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.003

  th.patience = 3
  th.early_stop = True
  th.early_stop_metric = 'f1'
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = False
  th.save_model = True

  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}bs{}'.format(model_name, th.batch_size)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate(False)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

