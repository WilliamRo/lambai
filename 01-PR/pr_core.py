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
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
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

th.feature_type = 1
th.win_size = 512
th.win_num = 1
th.aug_config = '-'

th.train_indices = '1,2'
th.val_indices = '3'
th.test_indices = '4'
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.alpha = 0
th.suffix = ''
# th.edge_cut = 50

th.epoch = 50000
th.early_stop = True
th.save_model = True
th.patience = 5
th.sample_num = 3

th.print_cycle = 5
th.updates_per_round = 30
th.validation_per_round = 1
th.export_tensors_upon_validation = True

th.val_batch_size = 1
th.val_progress_bar = True
th.eval_batch_size = 1
th.evaluate_train_set = False
th.evaluate_val_set = False
th.evaluate_test_set = True

th.tic_toc = True


def activate():
  if not th.allow_activation: return

  th.input_shape = [None, None, 1 if th.feature_type in (1, 9) else 2]

  # This block solves shape issue such as DUC
  if th.use_duc: th.fix_input_size = True
  if th.fix_input_size:
    th.input_shape[0] = th.input_shape[1] = th.win_size
    th.non_train_input_shape = [1024, 1280, th.input_shape[-1]]

  # Build model
  assert callable(th.model)
  model: Predictor = th.model()
  assert isinstance(model, Predictor)
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Load data
  train_set, val_set, test_set = du.load_data()

  # Train or evaluate
  th.probe_cycle = th.updates_per_round * th.epoch_per_probe
  # th.note_cycle = th.probe_cycle

  # ! Uncomment the line below to export 'predicted angle'
  from tframe import context
  # context.tensors_to_export['predicted angle'] = model.outputs.tensor

  if th.train:
    model.train(train_set, validation_set=val_set, test_set=test_set,
                trainer_hub=th, probe=du.PhaseSet.probe)

    for i in th.group_indices: test_set.snapshot(model, i)

    # Dump note.misc
    # train_set.dump_package(model)
  else:
    train_set.snapshot(model, 0, over_trial=True)
    data_set = test_set
    if th.visualize_tensors:
      if th.eval_rotation:
        data_set = data_set.rotation_test(None, index=0, data_leak=True)
      else:
        data_set = data_set[0:20:5]

      model.visualize_tensors(data_set, max_tensors=None, max_channels=50,
                              visualize_kernels=th.visualize_kernels,
                              tensor_dict=th.tensor_dict)
    elif th.eval_rotation:
      data_set.rotation_test(model, variation_diagram=True, index=6)
      # test_set.rotation_test(model)
    else:
      # train_set.evaluate_model(model)
      # val_set.evaluate_model(model)
      # model.evaluate_model(data_set, batch_size=1)
      data_set.evaluate_model(model)

  # End
  model.shutdown()
  console.end()
