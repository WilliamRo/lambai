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

  # Section I
  model.add(m.Conv2D(64, kernel_size=3, strides=2))
  model.add(m.BatchNormalization())
  model.add(m.Activation('relu'))
  p1 = model.add(m.MaxPool2D(2, strides=2))

  model.add(m.Conv2D(64, kernel_size=3, strides=1))
  model.add(m.BatchNormalization())
  model.add(m.Activation('relu'))
  model.add(m.Conv2D(64, kernel_size=3, strides=1))
  model.add(m.ShortCut(p1, mode=m.ShortCut.Mode.SUM))

  model.add(m.BatchNormalization())
  model.add(m.Activation('relu'))

  # Section II
  model.add(m.Conv2D(128, kernel_size=3, strides=2))
  model.add(m.BatchNormalization())
  model.add(m.Activation('relu'))
  p1 = model.add(m.MaxPool2D(2, strides=2))

  model.add(m.Conv2D(64, kernel_size=3, strides=1))
  model.add(m.BatchNormalization())
  model.add(m.Activation('relu'))
  model.add(m.Conv2D(64, kernel_size=3, strides=1))
  model.add(m.ShortCut(p1, mode=m.ShortCut.Mode.SUM))

  model.add(m.BatchNormalization())
  model.add(m.Activation('relu'))
  # TODO


  return m.finalize(th, model, flatten=True)


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
  th.mark = '{}lr{}bs{}dp{}'.format(
    model_name, th.learning_rate, th.batch_size, th.dropout)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

