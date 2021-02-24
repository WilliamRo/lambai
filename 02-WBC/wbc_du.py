import tframe as tfr
from tframe.data.augment.img_aug import image_augmentation_processor

from wbc.bc_set import BloodCellSet
from wbc.bc_agent import BloodCellAgent


def load_data(data_dir, raw_data_dir, val_config='d-2', test_config='d-3',
              H=350, W=320, pad_mode='constant', data_config=None):
  """Load data, see BloodCellAgent.load
    Currently converting dense label to one-hot tensors is done during
    preprocessing before splitting dataset.
  """
  train_set, val_set, test_set = BloodCellAgent.load(
    data_dir, raw_data_dir, val_config, test_config, H=H, W=W,
    pad_mode=pad_mode, data_config=data_config,
    save_HW_data=tfr.hub.save_HW_data)
  assert isinstance(train_set, BloodCellSet)
  assert isinstance(val_set, BloodCellSet)
  assert isinstance(test_set, BloodCellSet)
  # Set batch_preprocessor for augmentation if required
  if tfr.hub.augmentation:
    train_set.batch_preprocessor = image_augmentation_processor
  return train_set, val_set, test_set


if __name__ == '__main__':
  from wbc_core import th

  th.raw_data_dir = r'C:\Users\William\Dropbox\Shared\Share_Xin_William\Without Template'
  th.save_HW_data = False
  # th.only_BT = True
  train_set, val_set, test_set = load_data(
    th.data_dir, th.raw_data_dir, data_config='d-1,2,3', pad_mode='constant',
    H=300, W=300)
  test_set.view()
