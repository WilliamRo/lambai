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
  """Typical WBC3100(D) detail:
    -------------------------------------------------------
     Donor         B-Cell     T-Cell   Granuloc   Monocyte
    =======================================================
     1/6 HS1           19        161        167        121
     2/6 HS2          115        102        114        133
     3/6 HS3          140        107        111        137
     4/6 HS5            0        182        249        242
     5/6 HS6          365          0          0          0
     6/6 WBC_Q          0        249        191        195
    -------------------------------------------------------
     Overall          639        801        832        828
    -------------------------------------------------------
     Totally 3100 images; Max HxW: 347x320
    -------------------------------------------------------
  """

  DATA_NAME = 'WBCAlpha'

  PROPERTIES = {
    pedia.classes: ['B-Cell', 'T-Cell', 'Granulocyte', 'Monocyte'],
    BloodCellSet.NUM_CLASSES: 4
  }


  @classmethod
  def load(cls, data_dir, raw_data_dir, val_config='d-2', test_config='d-3',
           H=350, W=320, **kwargs):
    """Load train_set, val_set and test_set according to configuration strings.
    test_set will be separated first, following val_set.
    Remaining data will form the train_set.

    Configuration string logic for val_config and test_config:
    (1) Separate by donor indices:
        d-id_1[,id_2,...,id_N], id_i \in (1, 2, ...) represents donor ID
        e.g., 'd-2,3'
    (2) Separate by image number (over each cell type)
        [!]c-[!]r-num, where '!' denotes 'not', 'c' denotes 'over classes',
                             'r' denotes 'random' and 'num' is image number
        e.g., '!c-r-100', 'c-!r-200'

    :param data_dir: directory to keep .tfdir files
    :param raw_data_dir: directory containing raw images organized by donors
    :param val_config: configuration string for validation set
    :param test_config: configuration string for test set
    :param H: image height for preprocessing
    :param W: image width for preprocessing
    :param kwargs: other parameters
    :return: train_set, val_set and test_set
    """
    # Define configuration string logic
    def _parse_config(config: str):
      options = config.split('-')
      assert len(options) > 1
      num_or_indices, random, over_classes = None, None, None
      if options[0] == 'd':
        # (1) Separate by donor IDs
        assert len(options) == 2
        num_or_indices = [int(s) - 1 for s in options[1].split(',')]
      elif options[0] in ('c', '!c'):
        # (2) Separate by image number
        assert len(options) == 3 and options[1] in ('r', '!r')
        random = options[1] == 'r'
        over_classes = options[0] == 'c'
        num_or_indices = int(options[2])
      else: raise ValueError(
        '!! Unknown option {} for dataset separation'.format(options[0]))
      return {'num_or_indices': num_or_indices, 'random': random,
              'over_classes': over_classes}

    # This method is fixed for now
    data_set = cls.load_as_tframe_data(data_dir, raw_data_dir, with_donor=True)
    data_set.preprocess(H, W)
    # Separate test_set
    test_set, train_val_set = data_set.separate(
      name1='test_set', **_parse_config(test_config))
    # Separate val_set
    val_set, train_set = train_val_set.separate(
      name1='val_set', name2='train_set', **_parse_config(val_config))
    # Display details and return
    data_sets = (train_set, val_set, test_set)
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
    data_dir = os.path.join(data_dir, 'Person' if with_donor else 'ALL')
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
  raw_data_root = r'C:\Users\William\Dropbox\Shared\Share_Xin_William\Without Template'
  data_dir = os.path.abspath(__file__)
  for _ in range(2): data_dir = os.path.dirname(data_dir)
  data_dir = os.path.join(data_dir, 'data')
  val_config = 'c-!r-100'
  test_config = 'd-3'
  train_set, val_set, test_set = BloodCellAgent.load(
    data_dir, raw_data_root, val_config, test_config)

  # Show in image viewer
  # data_set.view()
