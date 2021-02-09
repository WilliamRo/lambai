import wbc_core as core
import wbc_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'alex'
id = 0
def model(th):
  """ Full version should be
  input[350x320] => reshape(350x320x1)
    => conv2d(3x3x64)->relu -> maxpool(2x2>2x2)
    => conv2d(3x3x192)->relu -> maxpool(2x2>2x2)
    => conv2d(3x3x384) -> relu
    => conv2d(3x3x256) -> relu
    => conv2d(3x3x256) -> relu -> maxpool(3x3>2x2) -> flatten(450560)
    => (dense(2048)->relu)x2 => dense(4) -> softmax => output[4]
  """
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False, add_last_dim=True)

  # Build classic AlexNet
  model.add(m.Conv2D(8, kernel_size=3, strides=1, activation='relu'))
  model.add(m.MaxPool2D(2, strides=2))

  model.add(m.Conv2D(16, kernel_size=3, strides=1, activation='relu'))
  model.add(m.MaxPool2D(2, strides=2))

  for filters in (32, 24, 24):
    model.add(m.Conv2D(filters, kernel_size=3))
    model.add(m.Activation('relu'))

  model.add(m.MaxPool2D(3, strides=2))
  model.add(m.Flatten())

  for dim in (128, 128):
    if th.dropout > 0: model.add(m.Dropout(1. - th.dropout))
    model.add(m.Dense(dim, activation='relu'))

  return m.finalize(th, model)


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
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

  th.dropout = 0.3

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 32
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.003

  th.patience = 5
  th.early_stop = True
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True

  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}'.format(model_name)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

