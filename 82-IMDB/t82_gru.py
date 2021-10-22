# -----------------------------------------------------------------------------
# Last modification: 2021-04-20
# -----------------------------------------------------------------------------
import imdb_core as core
import imdb_mu as m

import tensorflow as tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.gru import GRU
from tframe.utils.organizer.task_tools import update_job_dir

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gru'
id = 1
def model(th):
  assert isinstance(th, m.Config)
  cell = GRU(
    state_size=th.state_size,
    use_reset_gate=th.use_reset_gate,
    dropout_rate=th.rec_dropout,
  )
  return m.typical(th, cell)


def main(_):
  console.start('{} on IMDB task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.num_words = 5000
  th.max_len = 256

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.visible_gpu_id = 0
  th.suffix = '_t00'

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.hidden_dim = 100

  th.state_size = 128
  th.use_reset_gate = True

  th.input_dropout = 0.
  th.rec_dropout = 0.
  th.output_dropout = 0.
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.batch_size = 128

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  th.val_progress_bar = True
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  # th.export_states = True
  # th.export_gates = True

  # th.export_masked_weights = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.state_size)

  def add_dd(token, val):
    if val > 0: th.mark += '{}{}'.format(token, val)
  add_dd('i', th.input_dropout)
  add_dd('r', th.rec_dropout)
  add_dd('o', th.output_dropout)

  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


""" [Default] lr=0.001, state_size=128, num_words=10000

model(128) # 1.09M
0.351-85.3/84.9 (10)
0.346-86.9/86.15 (11) rdp=0.25
0.327-87.2/86.844 (7) nw=5000


"""
