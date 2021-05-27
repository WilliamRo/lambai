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

from pr.pr_configs import PRConfig as HubClass

import pr_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = HubClass(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [None, None, 1]
th.radius = 80
th.truncate_at = 12.0

th.win_size = 512
th.aug_config = '-'

th.train_indices = '1,2'
th.val_indices = '3'
th.test_indices = '4'
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5
# th.export_tensors_upon_validation = True
th.sample_num = 3

th.print_cycle = 2
th.updates_per_round = 20
th.validation_per_round = 2

th.val_batch_size = 1
th.val_progress_bar = True
th.eval_batch_size = 1
th.evaluate_train_set = False
th.evaluate_val_set = False
th.evaluate_test_set = True


def activate():
  if not th.allow_activation: return

  th.input_shape = [None, None, 1 if th.feature_type == 1 else 2]

  # Load data
  train_set, val_set, test_set = du.load_data()

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Predictor)

  # Train or evaluate
  th.probe_cycle = th.updates_per_round * th.epoch_per_probe
  th.note_cycle = th.probe_cycle

  from tframe import context
  context.tensors_to_export['predicted angle'] = model.outputs.tensor

  if th.train:
    model.train(train_set, validation_set=val_set, test_set=test_set,
                trainer_hub=th, probe=du.PhaseSet.probe)
    train_set.snapshot(model, 0)
    val_set.snapshot(model, 0)
    # test_set.snapshot(model, 3)
  else:
    data_set = test_set
    if th.visualize_tensors:
      if th.eval_rotation:
        data_set = data_set.rotation_test(None, index=0, data_leak=True)
      else:
        data_set = data_set[0:20:5]

      model.visualize_tensors(data_set, max_tensors=10, max_channels=16,
                              visualize_kernels=th.visualize_kernels)
    elif th.eval_rotation:
      data_set.rotation_test(model, variation_diagram=True, index=6)
      # test_set.rotation_test(model)
    else:
      # train_set.evaluate_model(model)
      # val_set.evaluate_model(model)
      data_set.evaluate_model(model)

  # End
  model.shutdown()
  console.end()
