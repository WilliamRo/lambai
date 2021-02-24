import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console, SaveMode
from tframe import Classifier
from wbc.wbc_hub import WBCHub

import wbc_du as du


from_root = lambda path: os.path.join(ROOT, path)

# ----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = WBCHub(as_global=True)
th.data_dir = from_root('02-WBC/data')
th.job_dir = from_root('02-WBC')
# -----------------------------------------------------------------------------
# Device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30
# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.image_side_length = 300
th.save_HW_data = True
th.pad_mode = 'constant'
th.num_classes = du.BloodCellAgent.PROPERTIES[du.BloodCellSet.NUM_CLASSES]

th.with_donor = True

th.val_config = 'd-2'
th.test_config = 'd-3'

th.augmentation = True
th.aug_config = 'flip|rotate'
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.show_structure_detail = True
# th.take_down_confusion_matrix = True  # evaluate method will handle it

th.early_stop = True
th.patience = 5
th.shuffle = True

th.save_model = True
th.save_mode = SaveMode.ON_RECORD
th.overwrite = True
th.gather_note = True

th.print_cycle = 4
th.validation_per_round = 2
th.export_tensors_upon_validation = True

th.val_batch_size = 128
th.eval_batch_size = 128
th.eval_metric = 'f1'
th.early_stop_metric = 'accuracy'

th.evaluate_train_set = False
th.evaluate_val_set = False
th.evaluate_test_set = False


def activate(visualize_false=False):
  # Input shape should be determined here
  th.input_shape = [th.image_side_length] * 2
  if th.only_BT: th.num_classes = 2

  # Load data (had been preprocessed)
  train_set, val_set, test_set = du.load_data(
    th.data_dir, th.raw_data_dir, th.val_config, th.test_config,
    pad_mode=th.pad_mode, H=th.image_side_length, W=th.image_side_length,
    data_config=th.data_config)

  if th.centralize_data:
    th.data_mean = train_set.feature_mean
    # th.data_std = train_set.feature_std

  # Build model
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Train or evaluate
  if th.train: model.train(
    train_set, validation_set=val_set, trainer_hub=th, test_set=test_set,
    evaluate=train_set.evaluate)
  else:
    # Evaluate model
    model.evaluate_pro(train_set, batch_size=th.eval_batch_size, verbose=True)
    model.evaluate_pro(val_set, batch_size=th.eval_batch_size, verbose=True)
    _, false_set = model.evaluate_pro(
      test_set, batch_size=th.eval_batch_size, verbose=True,
      show_confusion_matrix=True, show_class_detail=True, export_false=True)
    # Visualize false set
    if visualize_false: false_set.view()

  # End
  model.shutdown()
  console.end()


"""Notes

Things to be considered to do WBC:

1. Data Set configuration
  1.1 Padding method. 
    1.1.1 constant 0
    1.1.2 symmetric         <- todo
  1.2 Image shape
    1.1.1 300x300
    1.1.2 320x320
      seems to be promising
  1.3 Augmentation method
  1.4 Validation setting (early stop criterion)
    1.4.1 'd' style 
      seems to be better
    1.4.2 'c' style


"""
