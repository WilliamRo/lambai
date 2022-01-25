import pr_core as core
import pr_mu as m

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

from tframe.layers.hyper.dual_conv import DualConv2D
from pr.architectures.gad import Gad


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gad'
id = 8
def model():
  th = core.th

  # Create a Naphtali and modify model mark
  gad = Gad()
  model = m.get_container()

  # Add dual layer at the top if required
  if th.use_dual_conv:
    model.add(DualConv2D(m.get_hyper_channels(), th.prior_size,
                         filter_generator=m.konjac.dual_base))

  # Add Naphtali
  gad.add_to(model)
  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  token = 'beta'
  th.data_setup(token)

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

  th.use_dual_conv = False
  th.dual_setup()

  th.kernel_size = 7
  th.dilations = 10
  th.archi_string = '32-32-24-24-8'
  th.activation = 'relu'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.trainer_setup(token)
  th.batch_size = 4
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = model_name
  if th.use_dual_conv: th.mark += '(DC)'
  th.mark += f'({token})'
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


"""
Best V-WMAE after 10 epochs: 0.023
"""
