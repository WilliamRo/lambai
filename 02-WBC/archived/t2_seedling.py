import wbc_core as core
import wbc_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'seedling'
id = 0
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False, add_last_dim=True)

  # Parse archi_string
  nums = [int(s) for s in th.archi_string.split('-')]

  # Build classic AlexNet
  model.add(m.Conv2D(nums[0], kernel_size=3, strides=1, activation='relu'))
  model.add(m.MaxPool2D(2, strides=2))

  model.add(m.Conv2D(nums[1], kernel_size=3, strides=1, activation='relu'))
  model.add(m.MaxPool2D(2, strides=2))

  for filters in nums[2:]:
    model.add(m.Conv2D(filters, kernel_size=3))
    if th.use_batchnorm: model.add(m.BatchNormalization())
    model.add(m.Activation('relu'))

  if 'gap' in th.developer_code: model.add(m.GlobalAveragePooling2D())
  else: model.add(m.MaxPool2D(3, strides=2))
  model.add(m.Flatten())

  # Add re-thinker structure
  index_group = [0, 1]
  rth = m.Rethinker(index_group)
  fm = model.add_forkmerge(rth, name='wiseman')
  for branch_key in ('main', 'branch'):
    for dim in (64, 32):
      fm.add_to_branch(branch_key, m.Dense(dim, activation='relu'))
    # Add softmax units
    num = th.num_classes if branch_key == 'main' else len(index_group)
    fm.add_to_branch(branch_key, m.Dense(num))
    # Only add softmax to main branch
    if branch_key == 'main':
      fm.add_to_branch(branch_key, m.Activation('softmax'))

  return m.finalize(th, model, False, False)


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.image_side_length = 300

  th.centralize_data = True

  th.data_config = 'x'
  th.val_config = 'd-2'
  th.test_config = 'd-3'

  th.augmentation = True
  th.aug_config = 'flip|rotate'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  th.gpu_memory_fraction = 0.37
  th.allow_growth = False
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.archi_string = '8-16-24-18'
  th.developer_code = ''

  th.use_wise_man = True
  th.use_batchnorm = False

  th.dropout = 0.0
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 32

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0003

  th.patience = 5
  th.early_stop_metric = 'f1'

  th.train = True
  # ---------------------------------------------------------------------------
  # 4. display, summary and note setup
  # ---------------------------------------------------------------------------
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})_bs({})_val({})_tes({})'.format(
    model_name, th.archi_string, th.batch_size, th.val_config, th.test_config)
  core.activate(True)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


"""
Best HPs:
  th.archi_string = '8-16-24-18'

"""

