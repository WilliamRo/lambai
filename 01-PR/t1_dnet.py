import pr_core as core
import pr_mu as m

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir
from tframe.layers.hyper.dual_conv import DualConv2D

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'dnet'
id = 2

def model() -> core.Predictor:
  th = core.th
  model = m.get_container()

  # Add dual layer
  model.add(DualConv2D(m.get_hyper_channels(), th.prior_size,
                       filter_generator=m.konjac.dual_base))

  # Add U-Net body
  m.get_unet().add_to(model)
  return m.finalize(model)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th

  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_setup('alpha')

  th.visualize_tensors = True
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.allow_growth = False
  th.gpu_memory_fraction = 0.4
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  # Dual-conv configs
  th.use_dual_conv = True
  th.use_prior = th.use_dual_conv
  th.prior_size = 25
  th.prior_key = 'dual'
  th.kon_rs_str = '0.5'
  th.kon_rs = [float(s) for s in th.kon_rs_str.split(',')]
  th.kon_omega = 10
  th.kon_rad = 0.9
  th.n2o = 10.0

  # U-Net configs
  th.unet_setup(f=8, cks=3, duc=False)
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.trainer_setup('alpha')
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = model_name + th.dual_suffix()
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
