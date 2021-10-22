import pr_core as core
import pr_mu as m

from pr.architectures.dan import Dan

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir
from pr.architectures.ye_unet import Ye_UNet2D


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'yenet'
id = 3
def model():
  th = core.th
  model = m.get_container()
  model.add(m.conv(th.filters, th.kernel_size))

  Ye_UNet2D(filters=th.filters, kernel_size=th.kernel_size,
            activation=th.activation, height=th.half_height,
            use_batchnorm=th.use_batchnorm).add_to(model)

  return m.finalize(model, squish=True)


def main(_):
  console.start('{} on 2D Phase Retrieval task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.feature_type = 1
  th.win_size = 512
  th.win_num = 1

  th.fn_pattern = '0[45]-'

  indices = '1,2'
  th.train_indices = indices
  th.val_indices = indices
  th.test_indices = indices

  th.train_config = 't10'
  th.val_config = th.train_config
  th.test_config = '-5t'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.random_rotate = False

  th.allow_growth = False
  th.gpu_memory_fraction = 0.9
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.filters = 16
  th.half_height = 4
  th.kernel_size = 3
  th.relu_leak = 0.2
  th.activation = 'lrelu:0.2'

  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.loss_string = 'wmae:0.0001'
  # th.loss_string = 'mae'

  th.batch_size = 8

  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  th.patience = 20
  th.early_stop = True
  th.save_model = True

  # Commander Center
  th.eval_rotation = False
  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.epoch_per_probe = 10

  th.train_probe_ids = '0'
  # th.val_probe_ids = '0'
  th.test_probe_ids = '0'

  # Making configuration string
  config_str = 'ft{}-{}'.format(th.feature_type, th.loss_string.split(':')[0])
  if th.random_rotate: config_str += '-rot'

  th.mark = '{}({})'.format(model_name, config_str)
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  # For test
  if 0:
    th.feature_type = 1

    th.prefix = '0610_'
    th.bottle_neck = False

    th.train = False
    th.feature_type = 1
    th.visualize_tensors = True

    th.eval_rotation = False

  # th.rehearse = True
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

