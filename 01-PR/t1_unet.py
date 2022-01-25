import pr_core as core
import pr_mu as m

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'unet'
id = 5
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

  # ---------------------------------------------------------------------------
  # -1. test
  # ---------------------------------------------------------------------------
  if 0:
    th.train = [{False}]
    th.feature_type = [{1}]

    th.prefix = [{'0622_'}]
    # th.bridges = [{'4'}]
    th.contraction_kernel_size = [{5}]

    th.visualize_tensors = True
    th.eval_rotation = False

    th.use_duc = [{True}]
    th.use_dual_conv = [{False}]

  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  # th.feature_type = 1
  # th.win_size = 512
  # th.win_num = 1
  #
  # th.fn_pattern = '05-'
  # th.fn_pattern = '0[45]-'
  #
  # indices = '1'
  # indices = '1,2'
  # th.train_indices = indices
  # th.val_indices = indices
  # th.test_indices = indices
  #
  # th.int_para_1 = -10
  # th.train_config = 'a{}'.format(th.int_para_1)
  # th.val_config = '{}a'.format(th.int_para_1)
  #
  # th.train_config = 't5'
  # th.val_config = th.train_config
  #
  # th.test_config = '-5t'
  # # th.val_config = th.test_config

  th.data_setup('alpha')
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
  th.kernel_size = 3
  th.unet_setup(f=th.filters, cks=th.kernel_size, duc=False, act='relu')

  th.bridges = 'a'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  th.trainer_setup('alpha')
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = ''
  th.mark = '{}{}'.format(model_name, tail)
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  # th.allow_activation = False
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

