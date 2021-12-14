import os

import numpy as np

from tframe import pedia, console
from tframe.data.base_classes import DataAgent

from typing import Optional

from fb.fb_set import FBSet



class FBAgent(DataAgent):

  @classmethod
  def load(cls):
    # Load data set
    data_set = cls.load_as_tframe_data()

    # Mutate according to th.developer_code
    data_set.mutate()

    return data_set


  @classmethod
  def load_as_tframe_data(cls):

    # Load data directly if .tfd file exists
    file_path = cls._get_file_path()
    if os.path.exists(file_path): return FBSet.load(file_path)

    # Otherwise, read raw data and create a new .tfd file
    images, boxes = cls.load_as_numpy_arrays()

    data_set = FBSet(images, name='FBSet')
    data_set.boxes = boxes

    # Save dataset
    console.show_status('Saving dataset ...')
    data_set.save(file_path)
    console.show_status('Dataset has been saved to {}'.format(file_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls):
    from fb_core import th

    # Randomly generate some boxes
    return FBSet.randomly_generate_samples(th.fb_data_size, shape=th.fb_shape)


  @classmethod
  def _get_file_path(cls):
    from fb_core import th
    fn = 'FB-ds{}-is{}-L({}-{})-nb({}-{})-{}-ov{}'.format(
      th.fb_data_size, th.fb_img_size, th.fb_min_size, th.fb_max_size,
      th.fb_min_boxes, th.fb_max_boxes, th.fb_shape, th.fb_max_ovlp)
    return os.path.join(th.data_dir, f'{fn}.tfd')



if __name__ == '__main__':
  from fb_core import th

  data_set = FBAgent.load()
  print()
