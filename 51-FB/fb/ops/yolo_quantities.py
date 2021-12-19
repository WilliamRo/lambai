import numpy as np
from tframe import tf
from tframe.core.quantity import Quantity



def get_metric(key):
  """This method if for metric slot"""
  from fb.fb_set import FBSet
  assert key in ('AP', 'APC')

  def kernel(_, tensor_pred: tf.Tensor): return tensor_pred

  def ap(preds, data: FBSet):
    from fb.ops.yolout import YOLOut

    # If preds is generated in batches, concatenate them first
    if isinstance(preds, list): preds = np.concatenate(preds, axis=0)
    pred_boxes = YOLOut.pred_converter(preds)

    # Use the same thresholds as in FBSet
    thresholds = np.arange(0.1, 1.0, 0.1)
    return FBSet.calculate_AP(data.boxes, pred_boxes, thresholds=thresholds,
                              with_confidence=key=='APC')

  return Quantity(kernel, name=key, post_processor=ap, lower_is_better=False)


def get_loss():

  def kernel(tensor_true: tf.Tensor, tensor_pred: tf.Tensor):
    """Input shapes:
      tensor_pred.shape = [None, S, S, B, 5]
      tensor_true.shape = [None, S, S, B', 4], here B' = max(ASSIGNED_TARGETS)

    The tricky part of this loss function is that the ground-truth tensor
    is DYNAMICALLY GENERATED based on the predicted tensor. More specifically,
    the ground-truth tensor is calculated based on the IoU score.
    """
    from fb_core import th

    delta = tf.abs(tensor_true - tensor_pred)

    # Apply lambdas
    coor = tf.ones_like(tensor_pred[:, :, :, :, :4]) * th.yolo_coor
    assert th.yolo_noob < 1.0
    noob = tf.ones_like(tensor_pred[:, :, :, :, -1:])
    noob = tf.cast(noob > 0, tf.float32) * (1.0 - th.yolo_noob) + th.yolo_noob
    yolo_lambda = tf.concat([coor, noob], axis=-1)
    loss = tf.reduce_sum(delta * yolo_lambda, axis=[1, 2, 3, 4])

    return loss

  return Quantity(kernel, tf_summ_method=tf.reduce_mean,
                  np_summ_method=np.mean, name='YOLoss')
