import os

import numpy as np

from tframe import pedia, console
from tframe.data.base_classes import DataAgent
from tframe.data.shadow import DataShadow

from typing import Optional

from scis.scis_set import SCISet



class SCISAgent(DataAgent):

  @classmethod
  def load(cls, data_dir):
    from scis_core import th

    # Load whole dataset
    data_set = cls.load_as_tframe_data(data_dir)

    # Get subset if specified
    if th.cell_type is not None:
      data_set = data_set.get_subset_of_type(th.cell_type)

    # Report and return
    data_set.report()
    return data_set


  @classmethod
  def load_as_tframe_data(cls, data_dir, **kwargs):
    from scis_core import th

    # Load data directly if .tfd file exists
    file_path = cls._get_file_path(data_dir)
    if os.path.exists(file_path): return SCISet.load(file_path)

    # Otherwise, read raw data and create a new .tfd file
    images, meta_list, cell_types = cls.load_as_numpy_arrays(th.data_dir)

    data_set = SCISet(np.stack(images), name='SCISet', cell_types=cell_types)
    data_set.data_dict[data_set.Keys.META] = meta_list

    # Save dataset
    console.show_status('Saving dataset ...')
    data_set.save(file_path)
    console.show_status('Dataset has been saved to {}'.format(file_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    import csv, cv2

    # Find and read train.csv file
    RAW_FOLDER_NAME = 'sartorius-cell-instance-segmentation'
    raw_root = os.path.join(data_dir, RAW_FOLDER_NAME)
    csv_path = os.path.join(raw_root, 'train.csv')

    # Read csv files to rows
    with open(csv_path, newline='') as csvfile:
      rows = list(csv.reader(csvfile, delimiter=','))

    # Pop head
    head = [h.lower() for h in rows.pop(0)]

    # Initialize data_dict and fill it in
    data_dict = {key: [] for key in head}
    data_dict[SCISet.Keys.SHADOW] = []

    console.show_status('Encapsulating data ...')
    total = len(rows)
    meta: Optional[SCISet.ImageMeta] = None
    images, meta_list, cell_types = [], [], set()
    for i, row in enumerate(rows):
      # Show progress
      if i % (total // 100) == 0: console.print_progress(i, total=total)

      # Create new image, and append to images if necessary
      if meta is None or row[0] != meta.id:
        if meta is not None: meta_list.append(meta)
        meta = SCISet.ImageMeta(row[0], *row[2:])
        cell_types.add(meta.cell_type)
        # Read image
        path = os.path.join(raw_root, 'train', meta.id + '.png')
        images.append(cv2.imread(path))

      # Register cell to current image meta
      meta.register_cell(rle=row[1])

    # Append last meta
    meta_list.append(meta)
    console.show_status(f'Successfully loaded {len(meta_list)} images.')

    # Return image list
    return images, meta_list, cell_types


  @classmethod
  def _get_file_path(cls, data_dir):
    fn = 'SCIS-train'
    return os.path.join(data_dir, f'{fn}.tfd')



if __name__ == '__main__':
  from scis_core import th
  # th.cell_type = 'a'

  # data_set = SCISAgent.load_as_tframe_data(th.data_dir)
  data_set = SCISAgent.load(th.data_dir)

  # data_set.visualize()

