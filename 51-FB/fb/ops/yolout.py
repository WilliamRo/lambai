import numpy as np

from tframe import tf
from tframe.layers.layer import Layer
from tframe.operators.apis.neurobase import NeuroBase

from fb.ops.yolo_helper import YOLOImage
from talos.tasks.detection.box import Box



class YOLOut(Layer, NeuroBase):
  full_name = 'yolout'
  abbreviation = 'yolout'

  is_nucleus = True


  @property
  def structure_tail(self):
    from fb_core import th
    S, B, D = th.yolo_S, th.yolo_B, th.yolo_D
    return f'(S{S}B{B}D{D})'


  def _link(self, x: tf.Tensor):
    from fb_core import th

    # Make sure that the features are flattened
    assert len(x.shape) == 2

    # Calculate the output without class probabilities
    S, B, D = th.yolo_S, th.yolo_B, th.yolo_D

    assert D == 5
    p = self.dense(S * S * B * 4, x, scope='P', use_bias=True)
    c = self.dense(S * S * B * 1, x, scope='C', use_bias=True)

    alpha = 1.0
    p = tf.nn.tanh(p * alpha)
    c = tf.nn.sigmoid(c * alpha)

    p_reshape = tf.reshape(p, shape=[-1, S, S, B, 4], name='position')
    c_reshape = tf.reshape(c, shape=[-1, S, S, B, 1], name='confidence')
    output = tf.concat([p_reshape, c_reshape], axis=-1, name='output')

    if 'dup' in th.developer_code:
      with tf.variable_scope('Dummy'):
        v = tf.get_variable('v', shape=[1, S, S, B, 5], dtype=tf.float32,
                            initializer=tf.zeros_initializer(), trainable=True)
        output = v + 0.0 * output
        return output

    return output


  # region: Hairy Stuff

  @staticmethod
  def pred_converter(preds: np.ndarray, return_dicts=False):
    """For YOLO-like region-proposal algorithms, each prediction directly from
    models is of shape [batch_size, S, S, B, D]. This method converts such a
    tensor to a list of lists of boxes, so that the output preds[i] is like
    [[r_min_1, r_max_1, c_min_1, c_max_1], ...]
    Basically, the original [0, 0, 0, 0] should have the same location and shape
    of the corresponding grid.
    """
    from fb_core import th
    # Sanity check
    assert np.max(abs(preds)) <= 1.0
    assert th.yolo_D == 5

    # Get image shape
    L, S = th.fb_img_size, th.yolo_S
    boxes_list, dict_list = [], []
    for pred in preds:  # For pred for each image
      boxes, box_dict = [], {}
      # For each grid
      for i, j in [(i, j) for i in range(S) for j in range(S)]:
        grid_row_c, grid_col_c = YOLOImage.get_center_by_ij(i, j)  # ruby-alpha
        box_dict[(i, j)] = []
        for dr, dc, h, w, c in pred[i, j]:

          # Calculate center first
          box_row_c, box_col_c = grid_row_c + dr * L, grid_col_c + dc * L
          box_h = YOLOImage.get_box_length_1D(L, S, i, h)
          box_w = YOLOImage.get_box_length_1D(L, S, j, w)
          # Here box_[hw] >= 1.0
          r_min, r_max = box_row_c - box_h / 2, box_row_c + box_h / 2 - 1
          c_min, c_max = box_col_c - box_w / 2, box_col_c + box_w / 2 - 1

          box = Box(*[int(np.round(v)) for v in (r_min, r_max, c_min, c_max)],
                    confidence=c, tag=f'{c:.2f}')
          boxes.append(box)
          box_dict[(i, j)].append(box)

      # Append boxes
      boxes_list.append(boxes)
      dict_list.append(box_dict)

    if return_dicts: return dict_list
    return boxes_list

  @staticmethod
  def set_converter():
    from fb_core import th
    # th.data_converter = YOLOut.data_set_converter  # Deprecated
    th.dynamic_ground_truth_generator = YOLOut.set_dynamic_targets
    th.target_shape = [None, th.yolo_S, th.yolo_S, None, th.yolo_D]

  @staticmethod
  def generate_ground_truth(pred_tensor, batch, return_delta=False):
    # Sanity check
    from fb_core import th
    from fb.fb_set import FBSet
    from fb.ops.yolo_helper import YOLOImage
    assert isinstance(batch, FBSet)

    # Convert model output to boxes arranged in grid
    dict_list = YOLOut.pred_converter(pred_tensor, return_dicts=True)

    # Initialize the ground-truth as model output, shape = [?, S, S, B, D]
    target_tensor = pred_tensor.copy()
    # Init all confidence to 0.0
    target_tensor[:, :, :, :, -1] = 0.0
    delta_list = []

    # Set target in a brutal-force way
    S = th.yolo_S
    for im_index, (target_boxes, pred_dict) in enumerate(
        zip(batch.boxes, dict_list)):
      # Arrange targets in grid_dict
      target_dict = YOLOImage(target_boxes).grid
      delta = []
      for i, j in [(i, j) for i in range(S) for j in range(S)]:
        targets = target_dict[(i, j)]
        if len(targets) == 0: continue
        preds = pred_dict[(i, j)]
        iou_matrix = Box.calc_match_matrix(targets, preds)

        # Find the best fit for every target
        # (not find the best fit for every prediction)
        available_indices = set(range(len(preds)))
        for ious, box in zip(iou_matrix, targets):
          # Find predictions for this `box` assigned to grid[(i, j)]
          ranking = list(reversed(np.argsort(ious)))
          for index in ranking:
            if index in available_indices:
              # Remove index from available_indices
              available_indices.discard(index)
              # Set the hook
              dr, dc, h, w = YOLOImage.convert_box_to_rchw(box, i, j)
              iou = ious[index]

              # (*) Calculate sum-absolute difference
              gt = np.array([dr, dc, h, w, iou])
              d = np.sum(np.abs(gt - pred_tensor[im_index, i, j, index]))
              delta.append(d)

              # Set target
              target_tensor[im_index, i, j, index] = gt
              # Go to next target
              break

      # (*) For current image, add-up all delta
      delta_list.append(sum(delta))

    if return_delta: return delta_list
    return target_tensor

  @staticmethod
  def set_dynamic_targets(model, batch):
    # Sanity check
    from fb.fb_set import FBSet
    from tframe import Predictor
    assert isinstance(model, Predictor) and isinstance(batch, FBSet)

    # Get prediction if not provided
    pred_tensor = model.predict(batch)
    # Set and return
    batch.targets = YOLOut.generate_ground_truth(pred_tensor, batch)
    return batch

  # region: Deprecated

  @staticmethod
  def data_set_converter(data_set, is_training: bool, report_B_prime=False):
    """Create targets in data_set according to YOLO setting.
       targets = [batch_size, S, S, B_prime, 4]
       TODO: this method has been deprecated
    """
    if not is_training: return data_set

    from fb.fb_set import FBSet
    assert isinstance(data_set, FBSet)

    # Generate target list first
    images = [YOLOImage(box_list) for box_list in data_set.boxes]
    max_tgt_nums = [im.max_targets for im in images]
    B_prime = max(max_tgt_nums)

    if report_B_prime:
      from roma import console
      console.show_status(f"For {data_set.name}, B' = {B_prime}")

    # Set targets here
    data_set.targets = np.stack(
      [im.get_ground_truth_tensor(B_prime) for im in images], axis=0)

    return data_set

  # endregion: Deprecated

  # endregion: Hairy Stuff


if __name__ == '__main__':
  from fb.fb_agent import FBAgent
  from fb_core import th

  th.yolo_S, th.yolo_B = 3, 2

  th.set_data('g')
  # th.developer_code += '-dup'

  # data_set = FBAgent.load()
  # data_set = YOLOut.data_set_converter(data_set, True, report_B_prime=True)

  # Sanity check for inverse mappings
  # preds = YOLOut.pred_converter(data_set.targets)
  # data_set.visualize(preds)

  import fb_du as du
  train_set, val_set = du.load_data()
  val_set = YOLOut.data_set_converter(val_set, True, report_B_prime=True)
  preds = YOLOut.pred_converter(val_set.targets)
  val_set.visualize(preds)
