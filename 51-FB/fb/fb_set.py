import numpy as np

from collections import OrderedDict
from typing import List
from tframe import console
from talos import Nomear
from talos.tasks.detection.box import Box

from tframe.data.dataset import DataSet
from roma import check_type



class FBSet(DataSet):

  # region: Properties

  @property
  def boxes(self):
    """boxes is a list of lists of boxes"""
    return self.data_dict['boxes']

  @boxes.setter
  def boxes(self, val):
    check_type(val, list, list)
    self.data_dict['boxes'] = val

  @property
  def init_feed_dict(self): return {'boxes': self.boxes}

  # endregion: Properties

  # region: Overriding

  def _check_data(self):
    pass

  # endregion: Overriding

  # region: Public Methods

  def visualize(self, preds=None, **kwargs):
    from fb.fb_viewer import FBViewer
    from fb_core import th
    v = FBViewer(self, preds, **kwargs)
    if preds is not None:
      v.op_tag = True
      console.show_status(f'Lower bound for confidence is {th.min_confidence}')
    v.show()

  @staticmethod
  def randomly_generate_a_sample(
      im_size: int = None, n_boxes: int = None,
      min_box_size: int = None, max_box_size: int = None, shape='square'):
    """boxes[i] = [r, c, h, w]"""
    from fb_core import th

    # Set parameters if not provided
    if im_size is None: im_size = th.fb_img_size
    if n_boxes is None:
      n_boxes = np.random.randint(th.fb_min_boxes, th.fb_max_boxes + 1)
    if min_box_size is None: min_box_size = th.fb_min_size
    if max_box_size is None: max_box_size = th.fb_max_size

    # Generate a random sample
    image, boxes = np.zeros(shape=[im_size, im_size], dtype=np.float), []
    for _ in range(n_boxes):
      h, w = np.random.randint(min_box_size, max_box_size + 1, 2)
      ri = np.random.randint(0, im_size - h + 1)
      ci = np.random.randint(0, im_size - w + 1)

      # Generate box
      counter = 0
      while counter < 10:
        box = Box(r_min=ri, r_max=ri + h - 1, c_min=ci, c_max=ci + w - 1)
        if len([b for b in boxes if b.iou_to(box) > th.fb_max_ovlp]) == 0: break
        counter += 1
      if counter == 10: continue
      boxes.append(box)

      # Create shape
      if shape == 'square':
        image[ri:ri+h, ci:ci+w] = np.random.rand()
      elif shape == 'ellipse':
        # TODO
        raise NotImplementedError
      else: raise KeyError(f'Unknown shape `{shape}`')

    return image, boxes

  @staticmethod
  def randomly_generate_samples(
      n_samples, im_size=None, min_boxes=None, max_boxes=None,
      min_box_size=None, max_box_size=None, shape='square', verbose=False):
    from fb_core import th

    # Set parameters if not provided
    if min_boxes is None: min_boxes = th.fb_min_boxes
    if max_boxes is None: max_boxes = th.fb_max_boxes

    if verbose: console.show_status('Generating random boxes ...')
    images, boxes = [], []
    for i, n_boxes in enumerate(
        np.random.randint(min_boxes, max_boxes + 1, n_samples)):
      if verbose and (i % 5 == 0 or i == n_samples - 1):
        console.print_progress(i, n_samples)
      image, boxes_i = FBSet.randomly_generate_a_sample(
        im_size, n_boxes, min_box_size, max_box_size, shape=shape)
      images.append(image)
      boxes.append(boxes_i)

    return np.stack(images, axis=0), boxes

  def mutate(self):
    from fb_core import th
    if 'dup' in th.developer_code:
      self.features = np.stack([self.features[0]] * self.size, axis=0)
      self.boxes = [self.boxes[0]] * self.size
      console.show_status('[Mutation] `dup` takes effect')

  # endregion: Public Methods

  # region: Probing

  def evaluate_model(self, model, visualize=False):
    from fb_core import th
    from tframe import Predictor
    assert isinstance(model, Predictor)

    # Feed self into model and get results
    preds = model.predict(self, batch_size=10)

    # Convert predictions to boxes, boxes[i] = [TODO]
    preds: list = th.pred_converter(preds)
    assert len(preds) == len(self.boxes)

    # Calculate AP for each image
    AP = self.calculate_AP(self.boxes, preds)
    console.show_status('AP (on {}) = {:.3f}'.format(self.name, AP))

    # Visualize if necessary
    if visualize: self.visualize(preds)

  @staticmethod
  def calculate_AP(target_boxes, pred_boxes, thresholds=None,
                   with_confidence=False):
    from fb_core import th

    if thresholds is None: thresholds = np.arange(0.5, 1.0, 0.05)
    APs = []
    for gt, pd in zip(target_boxes, pred_boxes):
      pd = [b for b in pd if b.confidence >= th.min_confidence]
      if len(pd) == 0:
        APs.append(0.0)
        continue
      check_type(gt, list, Box)
      check_type(pd, list, Box)
      APs.append(Box.calc_avg_precision(
        gt, pd, thresholds, multiply_confidence=with_confidence))
    return np.average(APs)

  # endregion: Probing



if __name__ == '__main__':
  from fb.fb_agent import FBAgent

  data_set = FBAgent.load_as_tframe_data()
  data_set.visualize()

