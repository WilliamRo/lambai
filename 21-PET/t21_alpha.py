import pet_core as core
import pet_mu as m

from tframe import tf
from tframe import console
from tframe.utils.misc import date_string

from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'alpha'
id = 1
def model():
  model = m.get_container()
  m.construct_alpha(model)
  return m.finalize(model)


def main(_):
  console.start('{} on PET task'.format(model_name.upper()))

  th = core.th
  # th.rehearse = True
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.pet_input_size = 64

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.pet_mean = 38
  th.pet_radius = 10
  # th.num_classes = 50

  th.meta_branch_code = '32'
  # th.by_pass_image = True

  th.archi_string = '16-32-64-128-64'
  # th.archi_string = '8-16-32-64-128'
  th.kernel_size = 3
  th.activation = 'relu'
  # th.use_batchnorm = True

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 5000

  th.batch_size = 64
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0003
  th.lr_decay_method = 'cosine'
  th.dropout = 0.4

  th.patience = 5
  th.early_stop = True
  th.print_cycle = 10
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True

  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = (f'{model_name}({th.archi_string}-k{th.kernel_size}'
             f'-{th.activation})')
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

