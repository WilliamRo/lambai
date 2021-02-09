from tframe import console
from tframe import pedia
from tframe.data.images.ire_img import IrregularImageSet
from tframe.utils.misc import convert_to_one_hot
from tframe.utils.np_tools import pad_or_crop

import numpy as np


class BloodCellSet(IrregularImageSet):

  DONOR_KEY = 'donors'
  DONOR_NAME_KEY = 'donor_list'
  DENSE_LABEL_KEY = 'dense_labels'

  # region: Properties

  @property
  def donor_names(self): return self.properties[self.DONOR_NAME_KEY]


  @property
  def donors(self): return self.data_dict[self.DONOR_KEY]
  
  
  @property
  def donor_groups(self):
    if self.DONOR_KEY not in self.data_dict: return None
    groups = []
    for i, _ in enumerate(self.donor_names):
      groups.append(list(np.argwhere(self.donors == i).ravel()))
    return groups


  @property
  def dense_labels(self):
    if self.DENSE_LABEL_KEY not in self.data_dict: return self.targets
    else: return self.data_dict[self.DENSE_LABEL_KEY]


  @dense_labels.setter
  def dense_labels(self, val):
    self.data_dict[self.DENSE_LABEL_KEY] = val

  # endregion: Properties

  # region: Public Methods

  def split_by_donor(self, val_d_indices, test_d_indices):
    if isinstance(val_d_indices, int): val_d_indices = (val_d_indices,)
    if isinstance(test_d_indices, int): test_d_indices = (test_d_indices,)
    val_d_indices, test_d_indices = set(val_d_indices), set(test_d_indices)
    # Make sure val_set does not overlap with test_set
    assert len(val_d_indices.intersection(test_d_indices)) == 0
    # Find train_indices
    total_set = set(range(len(self.donor_names)))
    train_d_indices = total_set.difference(val_d_indices.union(test_d_indices))

    # Find image indices
    donor_groups = self.donor_groups
    indices_list = []
    for donor_indices in (train_d_indices, val_d_indices, test_d_indices):
      indices_list.append(np.concatenate(
        [np.array(donor_groups[donor_id]) for donor_id in donor_indices]))
    # Split and return
    datasets = self.split_by_indices(indices_list)
    return datasets


  def split_by_cell_type(self, val_size, test_size):
    # Sanity check, make sure train_set contains each class
    max_size = np.min([len(g) for g in self.groups])
    assert val_size + test_size < max_size
    # Find indices
    train_indices, val_indices, test_indices = [], [], []
    for g in self.groups:
      val_n_test_size = val_size + test_size
      train_indices += g[:-val_n_test_size]
      val_indices += g[-val_n_test_size:-test_size]
      test_indices += g[-test_size:]
    # Split and return
    return self.split_by_indices([train_indices, val_indices, test_indices])


  def split_by_indices(
      self, indices_list, names=('train_set', 'val_set', 'test_set')):
    assert len(indices_list) == len(names)
    data_sets = ()
    for indices, name in zip(indices_list, names):
      data_set = self[indices]
      data_set.name = name
      data_set.refresh_groups()
      data_sets += (data_set,)
    return data_sets


  def preprocess(self, H=350, W=320, pad_mode=0):
    from tframe.utils.display.progress_bar import ProgressBar
    assert pad_mode == 0
    # Preprocess one by one
    bar = ProgressBar(total=self.size)
    console.show_status('Preprocessing ...')
    for i, x in enumerate(self.features):
      # Shift each image so the min(image) is 0
      x -= np.min(x)
      # :: Pad or crop image
      # (1) Height
      x = pad_or_crop(x, axis=0, size=H)
      # (2) Width
      x = pad_or_crop(x, axis=1, size=W)
      self.features[i] = x
      # Show progress bar
      bar.show(i + 1)
    self.features = np.stack(self.features)
    # Convert label to one-hot tensors
    console.show_status('Converting dense labels to one-hot tensors ...')
    self.dense_labels = self.targets
    self.targets = convert_to_one_hot(self.targets, self.num_classes)
    console.show_status('Preprocess completed.')

  # endregion: Public Methods

  # region: Display

  def view(self, shuffle=True):
    from tframe.data.images.image_viewer import ImageViewer
    # Scale images to [0, 1]
    data_set = self[:]
    data_set.features = [x - np.min(x) for x in data_set.features]
    data_set.features = [x / np.max(x) for x in data_set.features]
    # Shuffle if necessary
    if shuffle:
      indices = list(range(self.size))
      np.random.shuffle(indices)
      data_set.features = [data_set.features[i] for i in indices]
      data_set.targets = data_set.targets[indices]
    # Show data using ImageViewer
    viewer = ImageViewer(data_set)
    viewer.show()


  def report_data_details(self):
    # For dataset with donor specification
    if self.DONOR_KEY in self.data_dict:
      console.show_info('Donor Detail in {}:'.format(self.name))
      donor_groups = self.donor_groups
      for donor_id, donor in enumerate(self.donor_names):
        if len(donor_groups[donor_id]) == 0: continue
        console.supplement('Donor {}/{} {}:'.format(
          donor_id + 1, len(self.donor_names), donor), level=2)
        for i, cell_name in enumerate(self.properties[pedia.classes]):
          # Get image # for each donor
          indices = [
            k for k in range(self.size)
            if self.dense_labels[k] == i and self.donors[k] == donor_id]
          console.supplement(
            '{} #: {}'.format(cell_name, len(indices)), level=3)

    console.show_info('Overall Detail of {}:'.format(self.name))
    # For dataset without donor specification
    for i, cell_name in enumerate(self.properties[pedia.classes]):
      console.supplement('{} #: {}'.format(
        cell_name, len(self.groups[i])), level=2)
    console.supplement('Totally {} images'.format(self.size))

    # Show size detail
    if isinstance(self.features, list):
      H = np.max([img.shape[0] for img in self.features])
      W = np.max([img.shape[1] for img in self.features])
      console.supplement('Max HxW: {}x{}'.format(H, W))
    else:
      assert isinstance(self.features, np.ndarray)
      shape = self.features.shape
      assert len(shape) == 3
      console.supplement('Image shape: {}x{}'.format(shape[1], shape[2]))

    console.split()


  # endregion: Display

  # region: Methods Overriding

  @classmethod
  def load(cls, filename):
    bcs = super().load(filename)
    assert isinstance(bcs, BloodCellSet)
    bcs.report_data_details()
    return bcs

  # endregion: Methods Overriding
