from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tframe as tfr

from tframe import pedia
from tframe import context
from tframe.layers.layer import Layer


class Rethinker(Layer):
  full_name = 'rethink'
  abbreviation = 'rethink'

  def __init__(self, *index_groups, loss_coef=1.0):
    # TODO: At stage I we allow only 1 group
    assert len(index_groups) == 1
    self.index_groups = index_groups
    self.loss_coef = loss_coef

    self.q_logits = None
    self.route_matrix = self.get_route_matrix()

    # If smart routing if turned off
    if not tfr.hub.use_wise_man:
      self.full_name = 'subconscious'
      self.abbreviation = 'subcons'


  def get_route_matrix(self):
    """Generate route matrix based on index group"""
    m = np.eye(tfr.hub.num_classes, dtype=float)
    for g in self.index_groups:
      assert isinstance(g, (list, tuple))
      m[np.ix_(g, g)] = 1
    return tf.constant(m, name='route_matrix', dtype=tf.float32)


  def _link(self, inputs:list, **kwargs):
    assert not self.linked
    assert len(inputs) == len(self.index_groups) + 1
    # Branch input should be logits
    p, q_logits = inputs
    assert all([isinstance(p, tf.Tensor), isinstance(q_logits, tf.Tensor)])

    # Inject error to q
    tfr.hub.show_extra_loss_info = True
    if self.loss_coef > 0:
      self.q_logits = q_logits
      context.customized_loss_f_net = self.branch_loss
    # Output directly if not use wise man
    if not tfr.hub.use_wise_man: return p

    # Get rethink probabilities
    q = tf.nn.softmax(q_logits)
    q_pad = tf.pad(q, [[0, 0], [0, 2]], 'CONSTANT')
    # :: Output while not training
    # Find max index
    p_one_hot = tf.one_hot(tf.argmax(p, axis=-1), depth=tfr.hub.num_classes)
    alpha = tf.matmul(p_one_hot, self.route_matrix)
    eval_output = alpha * p + (1 - alpha) * q_pad

    # Return
    is_training = tf.get_collection(pedia.is_training)[0]
    return tf.cond(is_training, lambda: p, lambda: eval_output)


  def branch_loss(self, model):
    """Borrow stuff from Net._get_customized_loss.
       This works only after target tensor of predictor having been plugged in.
    """
    from tframe import Predictor
    # Get necessary tensors
    q_logits = self.q_logits
    assert all([isinstance(model, Predictor), isinstance(q_logits, tf.Tensor)])
    targets = model._targets.tensor
    # Generate true labels in one-hot
    t = tf.stack([targets[:, i] for i in self.index_groups[0]], axis=-1)
    # Use weights to ensure that the branch is trained only when input is
    # .. of the corresponding types
    weights = tf.reduce_sum(t, axis=-1)

    # Calculate loss and return
    loss = tf.losses.softmax_cross_entropy(
      t, q_logits, weights, scope='branch_loss')
    return [loss * self.loss_coef]


if __name__ == '__main__':
  from tframe import console, hub
  console.suppress_logging()
  hub.num_classes = 6
  r = Rethinker((0, 1, 2), (3, 4))
  with tf.Session() as sess:
    console.eval_show(r.route_matrix)
