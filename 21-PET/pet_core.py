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
from tframe import Predictor
from pet.pet_configs import PetConfig as Hub

import pet_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.centralize_data = True

th.val_size = 2000
th.test_size = 2000

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5

th.print_cycle = 5
th.validation_per_round = 2

th.val_batch_size = 100
th.eval_batch_size = 100
th.val_progress_bar = True

th.evaluate_train_set = False
th.evaluate_val_set = False
th.evaluate_test_set = True

th.export_tensors_upon_validation = True


def activate():
  # Load data
  train_set, val_set, test_set = du.load_data(th.data_dir)
  if th.centralize_data: th.data_mean = train_set.feature_mean

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Predictor)

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Train or evaluate
  if th.train: model.train(
    train_set, validation_set=val_set, test_set=test_set, trainer_hub=th)
  else:
    model.evaluate_model(train_set, batch_size=th.eval_batch_size)
    model.evaluate_model(val_set, batch_size=th.eval_batch_size)
    model.evaluate_model(test_set, batch_size=th.eval_batch_size)

  # End
  model.shutdown()
  console.end()
