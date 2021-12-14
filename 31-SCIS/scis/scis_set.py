import numpy as np

from collections import OrderedDict
from typing import List
from tframe import console
from talos import Nomear
from tframe.data.dataset import DataSet
from talos.tasks.detection import RLEObject
from roma import check_type



class SCISet(DataSet):

  # region: Constants

  class Constants:
    thresholds = np.arange(0.5, 1.0, 0.05)

  # endregion: Constants

  # region: Subclasses

  class Keys:
    SHADOW = '_SHADOW_'
    META = '_META_'


  class Cell(RLEObject):

    def __init__(self, rle: str, meta):
      assert isinstance(meta, SCISet.ImageMeta)
      self.rle_str = rle

      # Parse RLE, so that rle[:, 0] is start, rle[:, 1] is length
      # TODO: error detected
      super().__init__([int(i) for i in rle.split()], width=meta.width)

    @Nomear.property()
    def rle2(self):
      """This should be overwritten since raw-data is problematic"""
      rle = np.reshape(self.rle, newshape=[-1, 2])
      rle[:, 1] = (rle[:, 0] + rle[:, 1] - 1)
      return rle - 1


  class ImageMeta(Nomear):
    def __init__(self, id: str, width: str, height: str, cell_type: str,
                 plate_time: str, sample_date: str, sample_id: str,
                 elapsed_timedelta: str):
      self.id = id
      self.width: int = int(width)
      self.height: int = int(height)
      self.cell_type = cell_type
      self.plate_time = plate_time
      self.sample_date = sample_date
      self.sample_id = sample_id
      self.elapsed_timedelta = elapsed_timedelta
      self.cells: List[SCISet.Cell] = []

    def register_cell(self, rle):
      self.cells.append(SCISet.Cell(rle, self))

    @Nomear.property()
    def mask(self) -> np.ndarray:
      mask = np.zeros(shape=[self.height, self.width, 3], dtype=bool)
      for i, cell in enumerate(self.cells):
        for r, c1, c2 in cell.rle3: mask[r, c1:c2+1, i % 3] = 1
      return mask

  # endregion: Subclasses

  # region: Properties

  @property
  def cell_types(self): return self['cell_types']

  @property
  def meta(self):
    meta_class = self.ImageMeta
    m: List[meta_class] = self.data_dict[self.Keys.META]
    return m

  @property
  def groups(self):
    g = {type_: [i for i, m in enumerate(self.meta) if m.cell_type == type_]
         for type_ in self.cell_types}
    return {k: v for k, v in g.items() if len(v) > 0}

  # endregion: Properties

  # region: Overriding

  def _check_data(self):
    pass

  # endregion: Overriding

  # region: Public Methods

  def get_subset_of_type(self, type_, return_indices=False):
    map = {'c': 'cort', 'a': 'astro', 's': 'shsy5y'}
    assert type_ in map.keys()
    type_ = map[type_]
    indices = self.groups[type_]
    if return_indices: return indices
    sub_set = self[indices]
    sub_set.name = f'{type_}-set'
    return sub_set

  def report(self):
    """Report data details"""
    H, W, _ = self.features[0].shape
    console.show_info(
      f'{self.name} includes {self.size} images of shape {H}x{W}:')
    for k, v in self.groups.items():
      num_cells = [len(self.meta[i].cells) for i in v]
      avg_cells = np.mean(num_cells)
      console.supplement(
        f"Type '{k}':"
        f" {len(v)} images, {min(num_cells)}-{max(num_cells)} cells, "
        f"{avg_cells:.1f} on average", level=2)

  def visualize(self):
    from .scis_viewer import SCISViewer

    sv = SCISViewer(self)
    sv.show()

  # endregion: Public Methods

  # region: Evaluation Methods

  @classmethod
  def _calculate_match_matrix(
      cls, true_list: List[RLEObject], pred_list: List[RLEObject],
      verbose=False):
    """Calculate the match matrix m of shape [M, N], where
       N = len(true_list), M = len(pred_list), and
       m[i, j] = iou(pred_list[i], true_list[j])
    """
    # Sanity check
    check_type(true_list, list, inner_type=RLEObject)
    check_type(pred_list, list, inner_type=RLEObject)

    # Create the matrix
    N, M = len(true_list), len(pred_list)
    m = np.zeros(shape=[M, N], dtype=float)
    if verbose:
      console.show_status(f'Calculating match matrix of size {M}x{N} ...')
    for i, pred_cell in enumerate(pred_list):
      if verbose and i % max(M // 50, 1) == 0: console.print_progress(i, M)
      for j, true_cell in enumerate(true_list):
        m[i, j] = pred_cell.iou_to(true_cell)

    if verbose: console.show_status('Done!')
    return m

  @classmethod
  def calculate_score(cls, true_list: list, pred_list: list, width=None,
                      verbose=False):
    """Calculate the competition metric for Sartorius challenge

    :param true_list: a list of ground-truth RLEs, for each i, true_list[i]
                      should be a RLE, e.g., [20, 8, 50, 6, ...]
    :param pred_list: a list of predicted RLEs
    :param width: image width, None by default
    :return: the score
    """
    # Convert RLE lists to RLEObjects if necessary
    if not isinstance(true_list[0], RLEObject):
      true_list = [RLEObject(rle, width) for rle in true_list]
      pred_list = [RLEObject(rle, width) for rle in pred_list]

    # Sanity check
    check_type(true_list, list, inner_type=RLEObject)
    check_type(pred_list, list, inner_type=RLEObject)

    # Sweep over all thresholds
    m = cls._calculate_match_matrix(true_list, pred_list, verbose=verbose)
    scores = []
    for threshold in cls.Constants.thresholds:
      TP = sum(np.sum(m >= threshold, axis=-1) > 0)
      FP, FN = m.shape[0] - TP, m.shape[1] - TP
      scores.append(TP / (TP + FP + FN))

    # Calculate precision and return
    precision = np.average(scores)
    return precision

  # endregion: Evaluation Methods


if __name__ == '__main__':
  from scis.scis_agent import SCISAgent
  from scis_core import th

  # print(SCISet.Constants.thresholds)

  data_set = SCISAgent.load_as_tframe_data(th.data_dir)

  # cell1 = data_set.meta[0].cells[0]
  # cell2 = data_set.meta[0].cells[0]
  # print(cell1.iou_to(cell2))

  index = 0
  true_list = data_set.meta[index].cells
  pred_list = data_set.meta[index + 0].cells
  print(SCISet.calculate_score(true_list, pred_list, verbose=True))

  # data_set.report()
  # data_set.visualize()


