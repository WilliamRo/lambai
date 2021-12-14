import numpy as np
import os

from collections import OrderedDict
from lambo.data_obj.interferogram import Interferogram
from lambo.analyzer.retriever import Retriever
from roma import console
from tframe.data.augment.img_aug import image_augmentation_processor
from tframe.data.base_classes import DataAgent
from roma import finder

from typing import Optional, List

from pr.pr_set import PhaseSet


class PRAgent(DataAgent):
  """Each pair of images should include the additional information below:
     (1) setup token
     (2) sample category
     Each folder in `data_dir` should contain `sample` and `bg` sub-folders
  """

  @classmethod
  def load(cls, data_dir, train_indices, val_indices, test_indices,
           radius: int, win_size: Optional[int] = None,
           truncate_at: float = 12.0, win_num: int = 1, feature_type: int = 1,
           fn_pattern='*[0-5][0-9]-*', re_pattern=None,
           **kwargs) -> List[PhaseSet]:
    """feature_type must be
          1: for raw interferogram
          2: for 2-D extracted image, real+imag
          3: for 2-D extracted image, amplitude+phase
    """

    from pr_core import th

    # Get some beta options
    random_rotate = kwargs.get('random_rotate', False)

    # Load complete dataset
    data_set = cls.load_as_tframe_data(
      data_dir, radius=radius, fn_pattern=fn_pattern, re_pattern=re_pattern,
      feature_type=feature_type)

    # Do some preprocess
    # if feature_type in (2, 3, 9): data_set.reset_feature(feature_type)
    data_set.add_channel()
    data_set.normalize_features()
    data_set.squash_target(truncate_at)
    if th.use_prior: data_set.set_prior()

    # Set window_postprocessor if required
    if win_size is not None:
      data_set.append_batch_preprocessor(
        PhaseSet.random_window_preprocessor(
          [win_size, win_size], win_num, random_rotate=random_rotate))

    # Check batch_preprocessor
    assert callable(data_set.batch_preprocessor)

    # Split datasets
    datasets = [data_set.get_subset_by_trial_indices(indices, name)
                for indices, name in zip(
        (train_indices, val_indices, test_indices),
        ('Train Set', 'Val Set', 'Test Set'))]

    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir: str, radius: int, feature_type: int,
                          fn_pattern='[0-5][0-9]-', re_pattern=None,
                          **kwargs) -> PhaseSet:

    # Check fn_pattern first
    fn_pattern = cls._check_fn_pattern(fn_pattern)
    # Load data directly if .tfd file exists
    file_path = os.path.join(data_dir, cls._get_file_name(
      radius=radius, data_dir=data_dir, fn_pattern=fn_pattern,
      re_pattern=re_pattern, feature_type=feature_type))
    if os.path.exists(file_path): return PhaseSet.load(file_path)

    # Load interferograms from given directory
    interferograms = cls.load_as_interferograms(
      data_dir, radius, fn_pattern, re_pattern)
    # Wrap them into PhaseSet
    console.show_status('Creating phase set ...')
    features, targets = [], []
    slopes, flatness = [], []
    for i, ig in enumerate(interferograms):
      assert isinstance(ig, Interferogram)
      features.append(ig.get_model_input(feature_type))
      targets.append(ig.flattened_phase)
      slopes.append(ig.bg_slope)
      flatness.append(ig.bg_flatness)

    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)

    # TODO: It is risky to put interferograms into `data_dict`
    ps = PhaseSet(name='PhaseSet', data_dict={
      'features': features, 'targets': targets,
      'interferograms': [ig.clone for ig in interferograms]})
    ps.data_dict[ps.SLOPE] = slopes
    ps.data_dict[ps.FLATNESS] = flatness

    # Save dataset
    console.show_status('Saving dataset ...')
    ps.save(file_path)
    console.show_status('Dataset has been saved to `{}`'.format(file_path))

    return ps


  @classmethod
  def load_as_interferograms(cls, data_dir: str, radius: int, fn_pattern: str,
                             re_pattern: Optional[str] = None):
    """Each folder in `data_dir` contains images and the corresponding
     backgrounds of a certain sample taken from one specific DPM system setup.
     `radius` denotes $k_0 \cdot NA$
    """
    # Initialize an empty list for interferograms
    interferograms = []

    # Search all folders inside given `data_dir`
    folder_paths = finder.walk(
      data_dir, type_filter='folder', pattern=fn_pattern, re_pattern=re_pattern)
    for k, folder_path in enumerate(folder_paths):

      # Get sample name
      sample_name = os.path.basename(folder_path)
      # Show status
      console.show_status(
        '[{}/{}] Scanning `{}` using lambo.Retriever ...'.format(
          k + 1, len(folder_paths), sample_name))

      # Load interferograms using lambo.Retriever
      interferograms.extend(Retriever.read_interferograms(
        folder_path, radius, seq_id=1, save=True,
        save_keys=('flattened_phase',)))

    console.show_status('{} interferograms read.'.format(len(interferograms)))
    return interferograms


  @classmethod
  def get_sample_bg_paths(cls, path: str):
    """Get List[(sample_path, bg_path)] given `path`. Files should be
    organized in one of the two ways:
    (1) The path of i-th sample is `path/sample/name[i].tif` and
        the path of its corresponding bg is `path/bg/name[i].tif`.
    (2) The path of i-th sample is 'path/<2*i-1>.tif' and the path of
        its corresponding bg is `path/<2*i>.tif`, where `<>` denotes
        evaluation.
    """
    pairs = []

    subfolders = finder.walk(path, type_filter='folder', return_basename=True)
    if 'sample' in subfolders and 'bg' in subfolders:
      # Case (1)
      sample_folder, bg_folder = [
        os.path.join(path, fn) for fn in ('sample', 'bg')]
      # Scan all sample files in sample folder
      for sample_path in finder.walk(
          sample_folder, type_filter='file', pattern='*.tif*'):
        fn = os.path.basename(sample_path)
        bg_path = os.path.join(bg_folder, fn)
        if not os.path.exists(bg_path): console.warning(
          ' ! Background file `{}` does not exist'.format(bg_path))
        else: pairs.append((sample_path, bg_path))
    else:
      # Case (2)
      file_list = finder.walk(
        path, type_filter='file', pattern='*.tif', return_basename=True)
      while len(file_list) > 0:
        fn = file_list.pop(0)
        # Get int id
        index = int(fn.split('.')[0])
        # If file is sample
        if index % 2 == 1:
          sample_fn = fn
          bg_fn = '{}.tif'.format(index + 1)
          mate = bg_fn
        else:
          bg_fn = fn
          sample_fn = '{}.tif'.format(index - 1)
          mate = sample_fn
        # Check if mate exists
        if mate not in file_list:
          console.warning(' ! Mate file `{}` of `{}` does not exist'.format(
            mate, fn))
          continue
        # Remove mate from list and append sample/bg path to pairs
        file_list.remove(mate)
        pairs.append(tuple([
          os.path.join(path, f) for f in (sample_fn, bg_fn)]))

    return pairs


  @classmethod
  def read_interferogram(cls, data_dir:str, trial_ID=1, sample_ID=1,
                         radius=80, pattern=None) -> Interferogram:
    if pattern is not None: pattern = cls._check_fn_pattern(pattern)
    trial_paths = finder.walk(data_dir, type_filter='folder', pattern=pattern)
    sample_bg_paths = cls.get_sample_bg_paths(trial_paths[trial_ID - 1])
    sample_path, bg_path = sample_bg_paths[sample_ID - 1]
    return Interferogram.imread(sample_path, bg_path, radius=radius)


  @classmethod
  def _get_file_name(cls, **kwargs):
    from lambo.misc.encode import encrypt_md5

    radius = kwargs.get('radius')
    data_dir = kwargs.get('data_dir')
    fn_pattern = kwargs.get('fn_pattern')
    re_pattern = kwargs.get('re_pattern')
    feature_type = kwargs.get('feature_type')
    token = '' if feature_type == 1 else '-ft{}'.format(feature_type)
    # Generate md5 suffix
    folders = finder.walk(data_dir, 'folder', return_basename=True,
                          pattern=fn_pattern, re_pattern=re_pattern)
    # suffix = encrypt_md5('='.join(folders) + token, digit=4)
    suffix = '({})'.format('-'.join([fn[:2] for fn in folders])) + token
    return 'PR-r{}-{}.tfd'.format(radius, suffix)


  @classmethod
  def _check_fn_pattern(cls, fn_pattern: Optional[str]):
    if fn_pattern is None: return None
    assert isinstance(fn_pattern, str)
    if fn_pattern[-1] != '*': fn_pattern = fn_pattern + '*'
    return fn_pattern


if __name__ == '__main__':
  from pr_core import th

  fn_pattern = '04'

  ps = PRAgent.load_as_tframe_data(
    th.data_dir, th.radius, feature_type=1, fn_pattern=fn_pattern)

  ps.view()



