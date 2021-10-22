import cf10_core as core
import cf10_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

from tframe import mu


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'seed'
id = 2
def model():
  th = core.th
  model = m.get_container()
  mu.NAS101(vertices=th.vertices.split(','), edges=th.adj_matrix,
            num_stacks=th.num_stacks, stem_channels=th.filters,
            cells_per_stack=th.module_per_stack,
            input_projection=th.input_projection).add_to(model)
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

  th.developer_code = '-'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_t00'

  th.visible_gpu_id = 0
  th.allow_growth = True
  th.gpu_memory_fraction = 0.8
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.vertices = 'conv3x3,maxpool3x3,conv3x3,conv1x1,conv3x3'
  th.adj_matrix = '1;01;001;1000;10011;100001'
  # th.vertices = 'b3'
  # th.adj_matrix = '1;11'
  th.filters = 64
  th.num_stacks = 3
  th.module_per_stack = 1

  th.input_projection = True
  th.use_batchnorm = True

  # th.global_l2_penalty = 0.0001
  # th.decoupled_l2_penalty = 0.000000001
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 500
  th.batch_size = 96
  th.val_batch_size = 100
  th.eval_batch_size = 100
  th.validation_per_round = 2

  th.optimizer = 'adam'
  th.learning_rate = 0.003

  th.patience = 10
  th.early_stop = True
  th.save_model = True

  # th.lives = 1
  # th.lr_decay = 0.4

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  # Make vertices and edge string shorter to avoid invalid checkpoint path issue
  vertices_str = ''.join(v[0] + v[-1] for v in th.vertices.split(','))
  edge_str = ''.join([str(int(n, 2)) for n in th.adj_matrix.split(';')])

  th.mark = '{}({}-{}-{}-{}-{})'.format(
    model_name, vertices_str, edge_str, th.filters, th.num_stacks,
    th.module_per_stack)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate(True)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

