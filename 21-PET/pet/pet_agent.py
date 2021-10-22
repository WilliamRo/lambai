import os
import sys

import numpy as np

from tframe import pedia, console
from tframe.data.base_classes import DataAgent
from tframe.data.dataset import DataSet
from tframe.data.shadow import DataShadow

from pet.pet_set import PetSet



class PetAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):

    data_set = cls.load_as_tframe_data(data_dir)

    # Split ...
    train_set, val_set, test_set = None, None, None
    return train_set, val_set, test_set


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs):
    """Load data as TFrame DataSet"""
    # Load data directly if .tfdi file exists
    file_path = os.path.join(data_dir, 'pet-2021.tfdir')
    if os.path.exists(file_path): return PetSet.load(file_path)

    # If not, read the raw data and create a new .tfdi file and save it
    data_dict = cls.load_as_numpy_arrays(data_dir)

    assert isinstance(data_dict, dict)
    data_set = PetSet(data_dict=data_dict, name='PET-2021')
    # Save data_set
    console.show_status('Saving dataset ...')
    data_set.save(file_path)
    console.show_status('Dataset has been saved to {}'.format(file_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    """Load (features, targets) as numpy arrays"""
    import csv
    from PIL import Image

    # Find and read train.csv file
    RAW_FOLDER_NAME = 'petfinder-pawpularity-score'
    raw_root = os.path.join(data_dir, RAW_FOLDER_NAME)
    csv_path = os.path.join(raw_root, 'train.csv')

    # Read csv files to rows
    with open(csv_path, newline='') as csvfile:
      rows = list(csv.reader(csvfile, delimiter=','))

    # Find head, and designate last column to be targets
    head = [h.lower() for h in rows.pop(0)]
    head[-1] = DataSet.TARGETS

    # Initialize data_dict and fill it in
    data_dict = {key: [] for key in head}
    data_dict[DataSet.FEATURES] = []

    console.show_status('Reading images ...')
    for i, row in enumerate(rows):
      console.print_progress(i, total=len(rows))

      img_path = os.path.join(RAW_FOLDER_NAME, 'train', row[0] + '.jpg')
      # Append shadow image to feature list
      data_dict[DataSet.FEATURES].append(DataShadow(img_path))

      # Fill in the metadata
      for k, v in zip(head, row):
        if k != 'id': v = int(v)
        data_dict[k].append(v)

    # Convert lists of regular elements to numpy arrays
    for k in head[1:]: data_dict[k] = np.array(data_dict[k])

    return data_dict


if __name__ == '__main__':
  from pet_core import th

  ps = PetAgent.load_as_tframe_data(th.data_dir)
  ps.visualize()

