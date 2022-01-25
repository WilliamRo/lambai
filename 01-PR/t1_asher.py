import pr_core as core
import pr_mu as m
from pr.architectures.asher import Asher

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'asher'
id = 9
def model() -> core.Predictor:
  model = m.get_container()

  asher = Asher()
  asher.add_to(model)

  return m.finalize(model, squish=False)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.win_size = 320

  token = 'beta'
  th.data_setup(token)
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.random_rotate = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.75
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.filters = 16
  th.kernel_size = 5
  th.dilations = 8
  th.thickness = 3

  th.activation = 'relu'
  th.ash_token = 'alpha'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.trainer_setup(token)
  th.batch_size = 8
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = f'(bs{th.batch_size}-{token})'
  th.mark = '{}{}'.format(model_name, tail)
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  # th.allow_activation = False
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

