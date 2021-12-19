import numpy as np
import matplotlib.pyplot as plt

from lambo import DaVinci
from scis.scis_set import SCISet
from talos.tasks.detection.box import Box



class SCISViewer(DaVinci):

  def __init__(self, data: SCISet, height=6, width=8):
    super(SCISViewer, self).__init__('Cell Viewer', height, width)
    self.data: SCISet = data

    # Initialize objects
    self.filter()

    # Add plotter
    self.add_plotter(self.get_show_image())

    # Options
    self.op_mask = False
    self.op_box = False

    # Register key events
    self._register_events_for_moving_rect(pixels=10)
    self.state_machine.register_key_event('m', self.toggle_mask)
    self.state_machine.register_key_event('b', self.toggle_box)

  # region: Plotters

  def get_show_image(self):

    def _show_image(x: int, ax: plt.Axes):
      im = self.data.features[x]
      # Show image
      ax.imshow(im)
      # Set title
      ax.set_title(f'Cell type: {self.data.meta[x].cell_type}')
      # Hide axis
      ax.set_axis_off()

      # Zoom-in to rect if provided
      self._zoom_in(ax)

      # Show mask if necessary
      if self.op_mask:
        # Plot mask naively
        ax.imshow(self.data.meta[x].mask.astype(np.float), alpha=0.2)

      # Show boxes if necessary
      if self.op_box:
        # Get boxes
        boxes = [c.box for c in self.data.meta[x].cells]
        for i, box in enumerate(boxes):
          box.show_rect(color=['r', 'g', 'b'][i % 3], ax=ax)

    return _show_image

  # endregion: Plotters

  # region: Commands

  def toggle_mask(self):
    self.op_mask = not self.op_mask
    self.refresh()

  def toggle_box(self):
    self.op_box = not self.op_box
    self.refresh()

  def filter(self, key=None, value=None):
    self._filter(key, value)
    self.refresh()

  def _filter(self, key, value):
    if key is None:
      self.objects = list(range(self.data.size))
    elif key in ('type', 'cell', 'cell_type', 't'):
      self.objects = self.data.get_subset_of_type(value, return_indices=True)

  # endregion: Commands

  # region: Public Methods

  # endregion: Public Methods


if __name__ == '__main__':
  from scis_core import th
  from scis.scis_agent import SCISAgent

  ss = SCISAgent.load_as_tframe_data(th.data_dir)
  ss.visualize()

