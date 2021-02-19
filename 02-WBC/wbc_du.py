from wbc.bc_set import BloodCellSet
from wbc.bc_agent import BloodCellAgent


def load_data(
    data_dir, raw_data_dir, val_config='d-2', test_config='d-3', H=350, W=320):
  """Load data, see BloodCellAgent.load
    Currently converting dense label to one-hot tensors is done during
    preprocessing before splitting dataset.
  """
  train_set, val_set, test_set = BloodCellAgent.load(
    data_dir, raw_data_dir, val_config, test_config, H=H, W=W)
  assert isinstance(train_set, BloodCellSet)
  assert isinstance(val_set, BloodCellSet)
  assert isinstance(test_set, BloodCellSet)
  return train_set, val_set, test_set


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  from wbc_core import th

  train_set, val_set, test_set = load_data(th.data_dir, None)
  # test_set.view()
