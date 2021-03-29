import numpy as np
import os
import time

from collections import OrderedDict
from lambo.data_obj.interferogram import Interferogram
from tframe import console
from tframe.data.base_classes import DataAgent
from tframe.utils.local import walk

from typing import Optional

from pr.pr_set import PhaseSet


class PRAgent(DataAgent):
  """Each pair of images should include the additional information below:
     (1) setup token
     (2) sample category
     Each folder in `data_dir` should contain `sample` and `bg` sub-folders
  """

  @classmethod
  def load(cls, data_dir, train_indices, val_indices, test_indices,
           radius: int, win_size: Optional[int] = None, **kwargs):
    # Load complete dataset
    data_set = cls.load_as_tframe_data(data_dir, radius=radius)
    data_set.add_channel()

    # Set batch_postprocessor if required
    if win_size is not None:
      data_set.batch_preprocessor = PhaseSet.random_window_preprocessor(
        [win_size, win_size])

    # Split datasets
    datasets = [data_set.get_subset_by_sample_indices(indices, name)
                for indices, name in zip(
        (train_indices, val_indices, test_indices),
        ('Train Set', 'Val Set', 'Test Set'))]

    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir: str, radius: int, **kwargs):
    # Load data directly if .tfd file exists
    file_path = os.path.join(data_dir, cls._get_file_name(radius=radius))
    if os.path.exists(file_path): return PhaseSet.load(file_path)

    # Load interferograms from given directory
    interferograms = cls.load_as_interferograms(data_dir, radius)
    # Wrap them into PhaseSet
    features = np.stack([ig.img for ig in interferograms], axis=0)
    # .. Calculate target one by one
    targets = []
    console.show_status('Retrieving phase ...')
    tic = time.time()
    for i, ig in enumerate(interferograms):
      assert isinstance(ig, Interferogram)
      targets.append(ig.flattened_phase)
      console.print_progress(i + 1, len(interferograms), start_time=tic)
    targets = np.stack(targets, axis=0)

    # TODO: It is risky to put interferograms into `data_dict`
    ps = PhaseSet(name='PhaseSet', data_dict={
      'features': features, 'targets': targets,
      'interferograms': interferograms})

    # Save dataset
    console.show_status('Saving dataset ...')
    ps.save(file_path)
    console.show_status('Dataset has been saved to `{}`'.format(file_path))

    return ps


  @classmethod
  def load_as_interferograms(cls, data_dir: str, radius: int):
    """Each folder in `data_dir` contains images and the corresponding
     backgrounds of a certain sample taken from one specific DPM system setup.
     `radius` denotes $k_0 \cdot NA$
    """
    interferograms = []

    for folder_path in walk(data_dir, type_filter='folder'):
      sample_name = os.path.basename(folder_path)
      sample_path, bg_path = [
        os.path.join(folder_path, fn) for fn in ('sample', 'bg')]
      console.show_status('Reading interferograms from `{}` ...'.format(
        sample_name))

      # Walk through the corresponding folder
      sample_file_paths = walk(sample_path, type_filter='file', pattern='*.tif')
      for i, sample_file_path in enumerate(sample_file_paths):
        # Check background image
        fn = os.path.basename(sample_file_path)
        bg_file_path = os.path.join(bg_path, fn)
        assert os.path.exists(bg_file_path)
        # Read sample and bg and do phase retrieval
        sample = Interferogram.imread(sample_file_path, radius=radius)
        bg = Interferogram.imread(bg_file_path, return_array=True)
        sample.set_background(bg)
        # Assign sample name and setup token
        sample.sample_token = sample_name
        sample.setup_token = '-'
        interferograms.append(sample)
        # Show progress bar
        console.print_progress(i + 1, len(sample_file_paths))

    console.show_status('{} interferograms read.'.format(len(interferograms)))
    return interferograms


  @classmethod
  def _get_file_name(cls, **kwargs):
    radius = kwargs.get('radius')
    return 'pr_r{}.tfd'.format(radius)


if __name__ == '__main__':
  from pr_core import th
  ps = PRAgent.load_as_tframe_data(th.data_dir, th.radius)
