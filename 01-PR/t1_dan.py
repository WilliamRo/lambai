import pr_core as core
import pr_mu as m

from pr.architectures.dan import Dan

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'dan'
id = 6
def model():
  th = core.th
  model = m.get_container()
  model.add(m.conv(th.filters, th.kernel_size))

  Dan(auto_mark=True).add_to(model)
  # tf.nn.conv2d

  return m.finalize(model, squish=True)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  token = 'beta'

  th.data_setup(token)
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.random_rotate = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.8
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.num_blocks = 5
  th.filters = 32
  th.kernel_size = 3
  th.dilations = 2
  th.relu_leak = 0

  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  # th.loss_string = 'wmae:0.0001'
  # # th.loss_string = 'mae'
  #
  # th.batch_size = 8
  #
  # th.optimizer = 'adam'
  # th.learning_rate = 0.0001
  #
  # th.patience = 20
  # th.early_stop = True
  # th.save_model = True

  # Commander Center
  th.eval_rotation = False
  th.train = True
  th.overwrite = True

  th.trainer_setup(token)
  th.batch_size = 4
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}'.format(model_name)
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  # th.rehearse = True
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

