import numpy as np

from talos.tasks.detection.box import Box



class YOLOImage(object):

  def __init__(self, boxes):
    from fb_core import th
    self.S = th.yolo_S
    self.L = th.fb_img_size
    self.boxes = boxes

    # Initialize grid
    self.grid = {(i, j): [] for i, j in [
      (i, j) for i in range(self.S) for j in range(self.S)]}

    # Assign each target to one box
    l = self.L // self.S
    for box in self.boxes:
      i, j = [int(c // l) for c in box.center]
      self.grid[(i, j)].append(box)


  @property
  def max_targets(self):
    return max([len(targets) for targets in self.grid.values()])


  @staticmethod
  def get_center_by_ij(i, j):
    from fb_core import th
    L, S = th.fb_img_size, th.yolo_S
    return Box.get_grid_center_1D(L, S, i), Box.get_grid_center_1D(L, S, j)


  @staticmethod
  def get_box_length_1D(L, S, i, ratio=0):
    """Inverse function of `get_ratio_1D`"""
    assert -1 <= ratio <= 1
    l = L // S
    if i == S - 1: l = L - l * (S - 1)
    if ratio >= 0: return l + (L - l) * ratio
    #  pixels:  1 ------ l
    #   ratio: -1 ...... 0
    return l + (l - 1) * ratio


  @staticmethod
  def get_ratio_1D(box_size, grid_size, im_size):
    """Inverse function of `get_box_length_1D`"""
    assert grid_size <= im_size
    if box_size >= grid_size:
      assert box_size <= im_size
      return (box_size - grid_size) / (im_size - grid_size)
    if grid_size == 1: return 0
    return (box_size - grid_size) / (grid_size - 1)


  @classmethod
  def convert_box_to_rchw(cls, box: Box, i, j,
                          grc=None, gcc=None, gh=None, gw=None):
    from fb_core import th
    S, L = th.yolo_S, th.fb_img_size

    # Check arguments
    if grc is None: grc, gcc = cls.get_center_by_ij(i, j)
    if gh is None:
      gh, gw = cls.get_box_length_1D(L, S, i), cls.get_box_length_1D(L, S, j)

    # Do calculation
    box_row_c, box_col_c = box.center
    dr, dc = (box_row_c - grc) / L, (box_col_c - gcc) / L

    # Calculate w, h in (-1, 1)
    h = cls.get_ratio_1D(box.height, gh, L)
    w = cls.get_ratio_1D(box.width, gw, L)

    return dr, dc, h, w


  def get_ground_truth_tensor(self, B_prime):
    """Convert boxes with integer-parameters to [-1, 1]x4 space"""
    from fb_core import th

    S = th.yolo_S
    tensor = np.zeros(shape=[S, S, B_prime, 5])
    # Fill-in tensor
    for i, j in [(i, j) for i in range(self.S) for j in range(self.S)]:
      grid_row_c, grid_col_c = self.get_center_by_ij(i, j)  # ruby-alpha
      grid_h = self.get_box_length_1D(self.L, self.S, i)
      grid_w = self.get_box_length_1D(self.L, self.S, j)
      for k, box in enumerate(self.grid[(i, j)]):
        dr, dc, h, w = self.convert_box_to_rchw(
          box, i, j, grid_row_c, grid_col_c, grid_h, grid_w)
        # Set tensor
        tensor[i, j, k] = [dr, dc, h, w, 1.0]

    return tensor



if __name__ == '__main__':
  from fb_core import th
  from fb.fb_agent import FBAgent
  from fb.ops.yolout import YOLOut

  th.yolo_S = 5
  th.yolo_B = 2

  data_set = FBAgent.load()
  data_set = YOLOut.data_set_converter(data_set, True, report_B_prime=True)

  # Sanity check for inverse mappings
  preds = YOLOut.pred_converter(data_set.targets)

  data_set.visualize(preds, pred_margin=1)

