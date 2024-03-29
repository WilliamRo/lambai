import pr_core as core
import pr_mu as m

from tframe import tf

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

  def add_conv(filters):
    return model.add(m.m.Conv2D(
      filters, th.kernel_size, use_bias=False, dilation_rate=th.dilations,
      use_batchnorm=th.use_batchnorm, activation=th.activation))

  for n in th.archi_string.split('-'):
    add_conv(int(n))

  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_setup('beta')

  th.visualize_tensors = True
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.allow_growth = False
  th.gpu_memory_fraction = 0.7
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.kernel_size = 5
  th.activation = 'relu'
  th.archi_string = '-'.join(['16'] * 25)
  th.archi_string = '16-32-64-64-64-32-16'
  th.dilations = 1

  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.trainer_setup('beta')

  th.batch_size = 4
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.suffix = ''
  if th.feature_type != 1: th.suffix += '_ft{}'.format(th.feature_type)
  if th.random_rotate: th.suffix += '_rot'

  tail = 'k{}-{}-{}'.format(th.kernel_size, th.archi_string, th.activation)
  if th.dilations != 1: tail += f'-dil{th.dilations}'
  th.mark = '{}({})'.format(model_name, tail)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  # th.rehearse = True
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

