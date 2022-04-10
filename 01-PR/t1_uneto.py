import pr_core as core
import pr_mu as m

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'uneto'
id = 3
def model() -> core.Predictor:
  th = core.th
  model = m.get_container()

  # Add U-Net body
  unet = m.get_unet()
  unet.add_to(model)
  model.mark += f'({unet})'
  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  th.developer_code = 'jan29'
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.win_size = 512

  # th.data_token = 'epsilon'
  th.data_token = 'eta'  # TODO
  th.train_config = '2a3'  # 2a3|4a5|39a40 for eta
  # th.train_config = '1a2'  # 0,3,7 for epsilon
  th.data_setup(th.data_token)
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_test10'

  th.random_rotate = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.75
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.filters = 8
  th.kernel_size = 3
  th.unet_setup(f=th.filters, cks=th.kernel_size, duc=False, act='relu')

  th.bridges = 'a'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  th.trainer_setup('beta')
  th.batch_size = 16

  th.patience = 10  # TODO
  # th.loss_string = 'mbe'
  th.loss_string = 'BER'
  # th.loss_string = 'f1'
  # th.loss_string = 'mae'
  # th.epoch_per_probe = 1

  # For model comparison
  th.validate_test_set = True
  # th.probe_cycle = [{1}]
  # th.epoch = 10
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = f'(bs{th.batch_size}-ws{th.win_size}-{th.data_token})'
  th.mark = '{}{}'.format(model_name, tail)
  th.mark += f'-{th.loss_string[:3]}-lr{th.learning_rate}'
  if 'xmae' in th.loss_string: th.mark += f'-a{th.alpha}'
  th.mark += f'-{th.train_config}'
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  # th.allow_activation = False
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

