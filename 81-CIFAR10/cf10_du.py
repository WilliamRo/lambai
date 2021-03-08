import tframe as tfr
from tframe import DataSet
from tframe.data.images.cifar10 import CIFAR10
from tframe.data.augment.img_aug import image_augmentation_processor



def load_data(path):
  train_set, val_set, test_set = CIFAR10.load(
    path, train_size=None, validate_size=5000, test_size=10000,
    flatten=False, one_hot=True)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  # Set batch_preprocessor for augmentation if required
  if tfr.hub.augmentation:
    train_set.batch_preprocessor = image_augmentation_processor
  return train_set, val_set, test_set


if __name__ == '__main__':
  from cf10_core import th
  from tframe.data.images.image_viewer import ImageViewer
  train_set, val_set, test_set = load_data(th.data_dir)
  viewer = ImageViewer(test_set)
  viewer.show()


