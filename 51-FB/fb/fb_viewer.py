import numpy as np
import matplotlib.pyplot as plt
from roma import Nomear

from lambo import DaVinci
from fb.fb_set import FBSet



class FBViewer(DaVinci):

  def __init__(self, data: FBSet, preds=None, height=7, width=7, **kwargs):
    super(FBViewer, self).__init__('Box Viewer', height, width)

    self.data: FBSet = data
    self.objects = list(range(self.data.size))

    # Check and set predictions
    if preds is not None: assert len(self.objects) == len(preds)
    self.preds = preds

    # Options
    self.op_box = False
    self.op_pred = True
    self.op_filter_pred = True
    self.op_tag = False
    self.S = 1

    # Register key events
    self._register_events()

    # Set kwargs
    self.kwargs = kwargs

    # Add plotters
    self.add_plotter(self.plot_boxes)
    self._add_pred_plotters()

  # region: Properties

  @property
  def confident_preds(self):
    if self.preds is None: return []
    from fb_core import th
    pred = self.preds[self.object_cursor]
    return [b for b in pred if b.confidence >= th.min_confidence]

  # endregion: Properties

  # region: Events

  def _move_obj_cursor(self, obj_cursor_shift: int):
    self.object_cursor += obj_cursor_shift
    self.layer_cursor = 0
    self._add_pred_plotters(obj_index=self.object_cursor)

  def _register_events(self):
    self.state_machine.register_key_event('b', self.toggle_box)
    self.state_machine.register_key_event('p', self.toggle_pred)
    self.state_machine.register_key_event('t', self.toggle_tag)
    self.state_machine.register_key_event('f', self.toggle_filter_pred)
    self.state_machine.register_key_event('[', lambda: self.shift_S(-1))
    self.state_machine.register_key_event(']', lambda: self.shift_S(1))

    # Overwrite key events
    self.state_machine.library['j'] = lambda: self._move_obj_cursor(1)
    self.state_machine.library['right'] = self.state_machine.library['j']
    self.state_machine.library['k'] = lambda: self._move_obj_cursor(-1)
    self.state_machine.library['left'] = self.state_machine.library['k']

  # endregion: Events

  # region: Plotters

  def plot_boxes(self, x: int, ax: plt.Axes, box_index=None):
    # Rename index
    i = x
    # Get image
    im = self.data.features[i]
    # Show image
    ax.imshow(im)
    # Hide axis
    ax.set_axis_off()

    # Plot boxes
    if self.op_box:
      for i, box in enumerate(self.data.boxes[i]):
        box.show_rect(['r', 'g', 'b'][i % 3], ax)

    # Show grid if necessary
    assert isinstance(self.S, int) and self.S > 0
    if self.S > 1:
      H, W = im.shape[:2]
      h, w = H // self.S, W // self.S
      alpha = 0.2
      for i in range(self.S - 1):
        x, y = w * (i + 1), h * (i + 1)
        # Horizontal
        ax.plot([0, W - 1], [y, y], 'w:', alpha=alpha)
        # Vertical
        ax.plot([x, x], [0, H - 1], 'w:', alpha=alpha)

    # Plot predictions
    if self.op_pred and self.preds is not None or box_index is not None:
      preds = self.confident_preds if self.op_filter_pred else self.preds[x]
      if box_index is not None: preds = [preds[box_index]]
      # Plot boxes
      for box in preds:
        box.show_rect('y', ax, show_tag=self.op_tag, linestyle=':')

  # endregion: Plotters

  # region: Private Methods

  def _add_pred_plotters(self, obj_index=0):
    if self.preds is None: return

    # Add plotter for each predicted box
    self.layer_plotters = [self.layer_plotters[0]]
    preds = self.preds[obj_index]
    if self.op_filter_pred: preds = self.confident_preds
    for i in range(len(preds)):
      self.add_plotter(lambda x, ax, i=i: self.plot_boxes(x, ax, box_index=i))

  # endregion: Private Methods

  # region: Commands

  def toggle_box(self):
    self.op_box = not self.op_box
    self.refresh()

  def toggle_tag(self):
    self.op_tag = not self.op_tag
    self.refresh()

  def toggle_pred(self):
    self.op_pred = not self.op_pred
    self.refresh()

  def toggle_filter_pred(self):
    self.op_filter_pred = not self.op_filter_pred
    self.layer_cursor = 0
    self._add_pred_plotters(obj_index=self.object_cursor)
    self.refresh()

  def shift_S(self, delta):
    assert delta in (-1, 1)
    if self.S == 1 and delta == -1: return
    self.S += delta
    self.refresh()

  def set_min_confidence(self, val: float):
    assert 0 <= val <= 1
    from fb_core import th
    th.min_confidence = val
    self.refresh()
  mc = set_min_confidence

  # endregion: Commands



if __name__ == '__main__':
  from fb_core import th
  from fb.fb_agent import FBAgent
  from fb.ops.yolout import YOLOut

  th.fb_data_size = 10
  # th.fb_shape = 'ellipse'

  # Case I
  # th.fb_img_size = 105

  # Case II
  th.fb_img_size = 54

  data_set = FBAgent.load_as_tframe_data()

  S, B, D = 5, 1, 5
  th.yolo_S, th.yolo_B, th.yolo_D = S, B, D
  preds = YOLOut.pred_converter(
    np.zeros(shape=[data_set.size, S, S, B, D], dtype=float))

  data_set.visualize(preds, pred_margin=0)

