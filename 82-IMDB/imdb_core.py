import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from tframe import Classifier
from tframe import DefaultHub as HubClass

import imdb_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = HubClass(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.35

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.num_words = 10000
th.max_len = 512
th.input_shape = [1]
th.num_classes = 2
th.output_dim = 1

th.train_size = 15000
th.val_size = 10000
th.test_size = 25000

# -----------------------------------------------------------------------------
# Set model configs
# -----------------------------------------------------------------------------
th.use_gather_indices = True
th.hidden_dim = 200

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.gather_note = True

th.batch_size = 256
th.num_steps = -1

th.epoch = 10000
th.early_stop = True
th.patience = 3
th.validation_per_round = 1

th.clip_threshold = 0.
th.clip_method = 'value'
th.val_batch_size = 1000
th.val_num_steps = -1

th.print_cycle = 1
th.val_progress_bar = True

th.evaluate_test_set = True
th.eval_batch_size = 2000

th.validation_per_round = 5
th.clip_method = 'value'
th.clip_threshold = 1.0

th.evaluate_train_set = True


def activate():
  # Load data
  train_set, val_set, test_set = du.load_data(
    th.data_dir, th.train_size, th.val_size, th.test_size, th.num_words,
    th.max_len)

  # Build model
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Train or evaluate
  if th.train:
    model.train(
      train_set, validation_set=val_set, test_set=test_set, trainer_hub=th)
  else:
    model.evaluate_model(train_set, batch_size=1)
    model.evaluate_model(val_set, batch_size=1)
    model.evaluate_model(test_set, batch_size=1)

  # End
  model.shutdown()
  console.end()
