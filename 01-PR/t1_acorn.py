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

  model.add(m.m.Conv2D(16, 3, activation='relu'))
  model.add(m.m.Conv2D(32, 3, activation='relu'))
  model.add(m.m.Conv2D(96, 3, activation='relu'))
  model.add(m.m.Conv2D(32, 3, activation='relu'))
  model.add(m.m.Conv2D(16, 3, activation='relu'))

  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.win_size = 256

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.pr_dev_code = 'dev.0'

  th.gpu_memory_fraction = 0.4
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 10
  th.validation_per_round = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.00003

  th.patience = 15
  th.early_stop = True
  th.save_model = True

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.print_cycle = 1

  th.mark = '{}_{}'.format(model_name, th.pr_dev_code)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

