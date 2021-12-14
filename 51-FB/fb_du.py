from fb.fb_agent import FBAgent
from fb.fb_set import FBSet



def load_data() -> [FBSet, FBSet]:
  from fb_core import th

  # Load data
  data_set = FBAgent.load()

  # Split data set
  train_set, val_set = data_set.split(
    -1, th.val_size, names=('Train Set', 'Val Set'))
  assert isinstance(train_set, FBSet) and isinstance(val_set, FBSet)

  # Deprecated
  # Convert data, set target if 'converter' is provided
  # if callable(th.data_converter):
  #   train_set.batch_preprocessor = th.data_converter

  return train_set, val_set



if __name__ == '__main__':
  from fb_core import th

  th.set_data('b')

  th.developer_code += '-dup'

  train_set, val_set = load_data()
  train_set.visualize()

