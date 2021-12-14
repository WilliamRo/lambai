import numpy as np
from tframe import tf
from tframe.core.quantity import Quantity



def get_metric():
  """This method if for metric slot"""
  from fb.fb_set import FBSet

  def kernel(_, tensor_pred: tf.Tensor): return tensor_pred

  def metric_post_processor(preds, data: FBSet):
    from fb.ops.yolout import YOLOut, Box

    # If preds is generated in batches, concatenate them first
    if isinstance(preds, list): preds = np.concatenate(preds, axis=0)
    pred_boxes = YOLOut.pred_converter(preds)

    # Use the same thresholds as in FBSet
    return FBSet.calculate_AP(data.boxes, pred_boxes)

  return Quantity(kernel, name='AP', post_processor=metric_post_processor,
                  lower_is_better=False)


def get_loss():

  def kernel(tensor_true: tf.Tensor, tensor_pred: tf.Tensor):
    """Input shapes:
      tensor_pred.shape = [None, S, S, B, 5]
      tensor_true.shape = [None, S, S, B', 4], here B' = max(ASSIGNED_TARGETS)

    The tricky part of this loss function is that the ground-truth tensor
    is DYNAMICALLY GENERATED based on the predicted tensor. More specifically,
    the ground-truth tensor is calculated based on the IoU score.
    """
    #

    delta = tf.abs(tensor_true - tensor_pred)
    loss = tf.reduce_sum(delta, axis=[1, 2, 3, 4])

    return loss

  return Quantity(kernel, tf_summ_method=tf.reduce_sum,
                  np_summ_method=np.sum, name='YOLoss')
