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
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_setup('alpha')

  th.visualize_tensors = True
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.allow_growth = False
  th.gpu_memory_fraction = 0.3
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.use_dual_conv = False
  th.dual_setup()

  th.kernel_size = 3
  th.archi_string = '8-8-4-4-2-2'

  # th.kernel_size = 3
  # th.num_blocks = 3
  # th.filters = 8
  # th.archi_string = '-'.join([str(th.filters) for _ in range(th.num_blocks)])

  th.dilations = 8

  th.activation = 'relu'
  th.input_projection = True
  th.nap_token = 'alpha'

  # Easy set
  if th.activation is not None:
    th.local_activation = th.activation
    th.global_activation = th.activation
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.trainer_setup('alpha')
  # th.batch_size = 1   # TODO: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = model_name
  if th.use_dual_conv: th.mark += '(DC)'
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()



