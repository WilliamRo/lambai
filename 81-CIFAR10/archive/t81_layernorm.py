import cf10_core as core
import cf10_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir
from tframe.nets.classic.conv_nets.lenet import LeNet


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'layernorm'
id = 3
def model():
  th = core.th
  model = m.get_container(flatten=False)
  lenet = LeNet(archi_string=th.archi_string)
  conv_list, fc_list = lenet.conv_list, lenet.fc_list

  # Add conv layers
  for i, n in enumerate(conv_list): model.add(m.Conv2D(
      n, th.kernel_size, strides=th.strides, activation=th.activation))

  # Add fc layers
  model.add(m.Flatten())
  for n in fc_list:
    model.add(m.Dense(n))
    assert not (th.layer_normalization and th.use_batchnorm)
    if th.layer_normalization: model.add(m.LayerNormalization())
    if th.use_batchnorm: model.add(m.BatchNormalization())
    model.add(m.Activation(th.activation))

  return m.finalize(model)


def main(_):
  console.start('{} on CIFAR-10 task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.centralize_data = True

  th.augmentation = True
  th.aug_config = 'flip:True;False'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_t00'

  th.visible_gpu_id = 0

  th.developer_code = '-'
  # th.developer_code = 'mg'
  # th.developer_code = ''
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '64-32=120-84'
  th.kernel_size = 4
  th.strides = 2
  th.activation = 'relu'

  th.use_batchnorm = True
  th.layer_normalization = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 500
  th.batch_size = 128
  th.validation_per_round = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.0003
  th.decoupled_l2_penalty = 0.0001

  # th.lr_decay_method = 'cos'

  th.patience = 5
  th.early_stop = True
  th.save_model = True

  th.train = True
  th.overwrite = True
  th.print_cycle = 20
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})_{}_s{}_k{}_dev-{}'.format(
    model_name, th.archi_string, th.activation, th.strides, th.kernel_size,
    th.developer_code)
  if th.use_batchnorm: th.mark += '_bn'
  elif th.layer_normalization: th.mark += '_ln'
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate(True)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

