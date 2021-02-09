import os
import numpy as np
import scipy.io as scio

from tframe import console
from tframe import pedia
from tframe.data.base_classes import DataAgent
from tframe.utils.local import walk

from tframe.utils.display.progress_bar import ProgressBar

from wbc.bc_set import BloodCellSet


class BloodCellAgent(DataAgent):

  DATA_NAME = 'WBCAlpha'

  PROPERTIES = {
    pedia.classes: ['B-Cell', 'T-Cell', 'Granulocyte', 'Monocyte'],
    BloodCellSet.NUM_CLASSES: 4
  }


  @classmethod
  def load(cls, data_dir, raw_data_dir, val_size_or_id=1,
           test_size_or_id=2, with_donor=True, H=350, W=320, **kwargs):
    # This method is fixed for now
    data_set = cls.load_as_tframe_data(data_dir, raw_data_dir, with_donor)
    data_set.preprocess(H, W)
    data_sets = data_set.split_by_donor(val_size_or_id, test_size_or_id)
    # Display details
    for ds in data_sets: ds.report_data_details()
    return data_sets


  @classmethod
  def load_as_tframe_data(
      cls, data_dir, raw_data_dir=None, with_donor=False, **kwargs):
    if raw_data_dir is None: raw_data_dir = data_dir
    # Load data directly if .tfdi file exists
    file_path = os.path.join(data_dir, cls._get_file_name(with_donor))
    if os.path.exists(file_path): return BloodCellSet.load(file_path)

    # Otherwise make a .tfdi file from raw data
    console.show_status('Generating irregular dataset from raw data...')
    if with_donor:
      images, labels, donor_id, donors_list = cls.load_as_numpy_arrays(
        raw_data_dir, with_donor=True, **kwargs)
      properties = {BloodCellSet.DONOR_NAME_KEY: donors_list}
      properties.update(cls.PROPERTIES)
      data_set = BloodCellSet(
        images, labels, name='WBCRaw(D)',
        data_dict={BloodCellSet.DONOR_KEY: donor_id}, **properties)
    else:
      images, labels = cls.load_as_numpy_arrays(
        raw_data_dir, with_donor=False, **kwargs)
      data_set = BloodCellSet(images, labels, name='WBCRaw', **cls.PROPERTIES)

    # Generate groups
    assert data_set.num_classes is not None
    data_set.refresh_groups()

    # Show status
    console.show_status('Successfully wrapped {} blood-cell images'.format(
      data_set.size))
    # Save data_set
    console.show_status('Saving dataset ...')
    data_set.save(file_path)
    console.show_status('Dataset has been saved to {}'.format(file_path))
    data_set.report_data_details()
    return data_set


  @classmethod
  def load_as_numpy_arrays(
      cls, data_dir, with_donor=False, mat_key='seg_phase_rtg'):
    if with_donor: return cls._load_cell4_with_donor(data_dir, mat_key)
    else: return cls._load_cell4_all(data_dir, mat_key)


  # region: Private Methods

  @classmethod
  def _read_classes(cls, data_dir, class_list, mat_key, allow_skip=False):
    """Load .mat data as 2-D numpy arrays.

    :param data_dir: data_dir should contain folders with names listed
      in class_list
    :param class_list: list of cell classes, the corresponding indices
      will be mapped to the dense label indices
    :return: Numpy image list and labels
    """
    # Sanity check
    assert isinstance(data_dir, str) and os.path.exists(data_dir)
    assert isinstance(class_list, (tuple, list)) and len(class_list) > 0

    # Read images
    X, Y = [], []
    for i, class_name in enumerate(class_list):
      # Set folder name
      folder_name = class_name
      if folder_name in ('T-Cell', 'B-Cell'):
        folder_name = folder_name.replace('C', 'c')
      # Find folder path
      folder_path = os.path.join(data_dir, folder_name)
      if not os.path.exists(folder_path):
        if allow_skip: continue
        raise FileNotFoundError(
          "!! Could not find folder '{}'".format(folder_path))

      # Find  all .mat files
      file_list = walk(folder_path, type_filter='file', pattern='*.mat')
      N = len(file_list)
      if N == 0: continue

      # Collect data
      console.show_status('Collecting {} data ...'.format(class_name))
      # Load .mat file one by one
      bar = ProgressBar(N)
      for j, fn in enumerate(file_list):
        bar.show(j)
        X.append(scio.loadmat(os.path.join(folder_path, fn))[mat_key])
      Y += [i] * N

      # Show number of loaded images
      console.supplement('{} images collected.'.format(N))

    return X, Y


  @classmethod
  def _load_cell4_all(cls, data_dir, mat_key):
    """Load all images from given data_dir. Images of the same type of
    cell should be organized in one folder. Images are assumed to be
    irregular. Thus the resulting X is a list of 2-D numpy array of
    different shapes."""
    X, Y = cls._read_classes(data_dir, cls.PROPERTIES[pedia.classes], mat_key)
    Y = np.array(Y)
    return X, Y


  @classmethod
  def _load_cell4_with_donor(cls, data_dir, mat_key):
    """Folders in 'data_dir' should be organized by donors"""
    X_all, Y_all, donor_indices, donors_names = [], [], [], []
    donor_paths = walk(data_dir, type_filter='dir')
    for d_path in donor_paths:
      donor = os.path.basename(d_path)
      console.section('Collecting data from donor {}'.format(donor))
      X, Y = cls._read_classes(d_path, cls.PROPERTIES[pedia.classes],
                               mat_key, allow_skip=True)
      if len(X) == 0: continue
      # Register donor
      donor_indices += [len(donors_names)] * len(X)
      donors_names.append(donor)
      # Append X and Y
      X_all += X
      Y_all += Y

    # Convert to numpy array if necessary and return
    Y_all = np.array(Y_all)
    donor_indices = np.array(donor_indices)
    return X_all, Y_all, donor_indices, donors_names


  @classmethod
  def _get_file_name(cls, with_donor):
    """Get .tfdir file name given setup details"""
    file_name = 'wbc-4-' + ('donor' if with_donor else 'all')
    return file_name + '.tfdir'

  # endregion: Private Methods


if __name__ == '__main__':
  with_donors = True
  raw_data_root = r'C:\Users\William\Dropbox\Shared\Share_Xin_William\Without Template'

  if with_donors: raw_data_root += r'\Person'
  else: raw_data_root += r'\All'

  data_dir = os.path.abspath(__file__)
  for _ in range(2): data_dir = os.path.dirname(data_dir)
  data_dir = os.path.join(data_dir, 'data')
  data_set = BloodCellAgent.load(
    data_dir, raw_data_root, with_donor=with_donors)


  # Show in image viewer
  # data_set.view()
