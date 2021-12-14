import os

import numpy as np

from tframe import pedia, console
from tframe.data.base_classes import DataAgent
from tframe.data.shadow import DataShadow

from pet.pet_set import PetSet



class PetAgent(DataAgent):

  MAX_SIZE_TO_SAVE = 256

  @classmethod
  def load(cls, data_dir):
    from pet_core import th

    data_set = cls.load_as_tframe_data(data_dir, traverse=True)

    # Modify data shadow if necessary
    if os.name == 'posix':
      for s in data_set.shadows:
        assert isinstance(s, DataShadow)
        s.data_path = s.data_path.replace('\\', '/')

    # Set targets
    data_set.set_targets()

    # Split data set accordingly
    if th.folds_k is None: data_sets = data_set.split(
      -1, th.val_size, th.test_size, names=('Train Set', 'Val Set', 'Test Set'))
    else: data_sets = data_set.split_k_fold(th.folds_k, th.folds_i)

    for ds in data_sets: ds.report()
    return data_sets


  @classmethod
  def load_as_tframe_data(cls, data_dir, traverse=True, **kwargs):
    """Load data as TFrame DataSet"""
    from pet_core import th

    # Set max length for DataShadow
    DataShadow.set_max_size(500)

    # Load data directly if .tfdi file exists
    file_path = cls.get_file_name(data_dir, traverse)
    if os.path.exists(file_path): return PetSet.load(file_path)

    # If not, read the raw data and create a new .tfdi file and save it
    data_dict = cls.load_as_numpy_arrays(data_dir)

    assert isinstance(data_dict, dict)
    data_set = PetSet(data_dict=data_dict, name='PET-2021')

    # Traverse data_set if necessary
    if traverse:
      console.show_status('Traversing pet-set ...')
      # Check L
      L = th.pet_input_size
      if L is not None and L > cls.MAX_SIZE_TO_SAVE:
        raise AssertionError(f'Image size should be no greater than '
                             f'{cls.MAX_SIZE_TO_SAVE} for saving')

      sizes, unified_imgs = [], []
      for i, s in enumerate(data_set.shadows):
        console.print_progress(i, data_set.size)
        im: np.ndarray = s.data
        sizes.append(im.shape[:2])
        if L is not None: unified_imgs.append(data_set.unify(im))
      data_set.data_dict[data_set.SIZES_KEY] = sizes

      # Set features if necessary
      if L is not None:
        data_set.data_dict[data_set.FEATURES] = np.stack(unified_imgs, axis=0)

    # Save data_set
    console.show_status('Saving dataset ...')
    data_set.save(file_path)
    console.show_status('Dataset has been saved to {}'.format(file_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    """Load (features, targets) as numpy arrays"""
    import csv

    # Find and read train.csv file
    RAW_FOLDER_NAME = 'petfinder-pawpularity-score'
    raw_root = os.path.join(data_dir, RAW_FOLDER_NAME)
    csv_path = os.path.join(raw_root, 'train.csv')

    # Read csv files to rows
    with open(csv_path, newline='') as csvfile:
      rows = list(csv.reader(csvfile, delimiter=','))

    # Find head, and designate last column to be targets
    head = [h.lower() for h in rows.pop(0)]
    head[-1] = PetSet.TARGETS

    # Initialize data_dict and fill it in
    data_dict = {key: [] for key in head}
    data_dict[PetSet.SHADOW_KEY] = []

    console.show_status('Creating image shadows ...')
    for i, row in enumerate(rows):
      console.print_progress(i, total=len(rows))

      img_path = os.path.join(RAW_FOLDER_NAME, 'train', row[0] + '.jpg')
      # Append shadow image to feature list
      data_dict[PetSet.SHADOW_KEY].append(DataShadow(img_path))

      # Fill in the metadata
      for k, v in zip(head, row):
        if k != 'id': v = int(v)
        data_dict[k].append(v)

    # Convert lists of regular elements to numpy arrays
    for k in head[1:]: data_dict[k] = np.array(data_dict[k])

    return data_dict


  @classmethod
  def get_file_name(cls, data_dir, traverse):
    from pet_core import th
    suffix = ''
    if traverse:
      L = th.pet_input_size
      suffix = '-x' if L is None else f'-{L}x{L}'
    file_path = os.path.join(data_dir, 'pet-2021' + suffix + '.tfdir')
    return file_path


if __name__ == '__main__':
  from pet_core import th

  ps = PetAgent.load_as_tframe_data(th.data_dir, True)
  ps.visualize()

