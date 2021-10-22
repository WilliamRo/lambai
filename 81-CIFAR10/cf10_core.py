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
from tframe import DefaultHub

import cf10_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = DefaultHub(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [32, 32, 3]
th.num_classes = 10

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5

th.print_cycle = 20
th.validation_per_round = 2

th.centralize_data = True

th.val_batch_size = 2000
th.eval_batch_size = 2000

th.evaluate_train_set = False
th.evaluate_val_set = False
th.evaluate_test_set = False

th.validate_train_set = True
th.validate_test_set = True

th.class_indices = '3,5'
th.export_tensors_upon_validation = True


def activate(visualize_false=False):
  # Load data
  train_set, val_set, test_set = du.load_data(th.data_dir)
  if th.centralize_data: th.data_mean = train_set.feature_mean

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  # Train or evaluate
  if th.train: model.train(
    train_set, validation_set=val_set, test_set=test_set, trainer_hub=th,
    evaluate=du.evaluate)
  else: model.evaluate_image_sets(
    train_set, val_set, test_set, visualize_last_false_set=visualize_false)

  # End
  model.shutdown()
  console.end()
