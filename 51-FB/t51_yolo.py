import fb_core as core
import fb_mu as m

from tframe import tf
from tframe import console
from tframe.utils.misc import date_string

from tframe.utils.organizer.task_tools import update_job_dir

from fb.ops.yolout import YOLOut
import fb.ops.yolo_quantities as yoloq


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'yolo'
id = 1
def model():
  YOLOut.set_converter()

  model = m.get_container()

  # Construct body
  # m.add_feature_extractor(
  #   model, channels=16, n_blocks=3, flatten=True)

  model.add(m.Flatten())

  model.add(YOLOut())

  # Build model
  model.build(loss=yoloq.get_loss(), metric=yoloq.get_metric(),
              batch_metric='AP')
  return model


def main(_):
  console.start('{} on FB task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.pred_converter = YOLOut.pred_converter

  token = 'b'
  th.set_data(token)

  th.developer_code += '-dup'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = '_demo'
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.kernel_size = 3

  th.yolo_S = 3
  th.yolo_B = 1

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000

  th.batch_size = 8
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  # th.optimizer = tf.train.GradientDescentOptimizer
  th.learning_rate = 0.00003
  # th.lr_decay_method = 'cos'

  th.patience = 5
  th.early_stop = True
  th.print_cycle = 1
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.visualize_after_training = True
  th.save_model = True

  th.overwrite = False
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = model_name + f'(S{th.yolo_S}B{th.yolo_B})-{token}'
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

