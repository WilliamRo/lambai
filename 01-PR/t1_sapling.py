import pr_core as core
import pr_mu as m
from pr.architectures.sapling import Naphtali
from tframe import console
from tframe import tf
from tframe.layers.hyper.dual_conv import DualConv2D
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'sapling'
id = 1
def model():
  th = core.th

  # Create a Naphtali and modify model mark # Hello
  naphtali = Naphtali()
  model = m.get_container()

  # Add dual layer at the top if required
  if th.use_dual_conv:
    model.add(DualConv2D(
      m.get_hyper_channels(), th.prior_size, filter_generator=m.konjac.dual_base))
    # model.add(m.m.BatchNorm())

  # Add Naphtali
  naphtali.add_to(model)
  return m.finalize(model)


def main(_):
  console.start('{} on 3D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # th.rehearse = True
  th.developer_code = 'jan29'
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_token = 'eta'
  th.train_config = '4a5'  # 2a3|4a5|39a40 for eta
  # th.train_config = 'a1'  # 0,1,2,3,4,5,7 for epsilon
  th.data_setup(th.data_token)

  th.visualize_tensors = True
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.allow_growth = True
  th.gpu_memory_fraction = 0.8
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.use_dual_conv = False
  th.dual_setup()

  th.kernel_size = 9
  th.archi_string = '32-32-24-16-8'

  th.dilations = 8

  th.activation = 'relu'
  th.input_projection = True
  th.nap_token = 'beta'

  # Easy set
  if th.activation is not None:
    th.local_activation = th.activation
    th.global_activation = th.activation
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True  #
  th.visualize_tensors = False

  th.trainer_setup('beta')
  th.batch_size = 12
  th.loss_string = 'ber'
  # th.loss_string = 'mbe'
  th.patience = 10  # TODO

  th.validate_test_set = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = model_name
  if th.use_dual_conv: th.mark += '(DC)'
  th.mark += f'(bs{th.batch_size}-{th.data_token})'
  th.mark += f'-{th.loss_string[:3]}'
  if 'xmae' in th.loss_string: th.mark += f'-a{th.alpha}'
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()



