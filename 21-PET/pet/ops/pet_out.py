import tframe as tfr
from tframe import tf
from tframe.layers.layer import Layer
from tframe.operators.apis.neurobase import NeuroBase



class PetOut(Layer, NeuroBase):
  full_name = 'pet_out'
  abbreviation = 'pet_out'

  is_nucleus = True


  @property
  def structure_tail(self):
    from pet_core import th

    tail = '(sigmoid)' if not th.use_classifier else f'(c-{th.num_classes})'
    return tail


  def _link(self, x: tf.Tensor):
    from pet_core import th

    # Case I
    if not th.use_classifier:
      y = self.dense_v2(1, 'output', x, activation='sigmoid')

      # Case I-i: Use conservative strategy
      assert 0 < th.pet_radius < 50
      if th.pet_radius < th.pet_mean < 100 - th.pet_radius:
        return 2 * (y - 0.5) * th.pet_radius + th.pet_mean

      # Case I-ii: Otherwise
      return 1.0 + y * 99

    # Case II
    th.check_num_classes()
    N = 100 // th.num_classes

    logits: tf.Tensor = self.dense_v2(th.num_classes, 'logits', x)
    y = tf.nn.softmax(logits)

    assert len(logits.shape) == 2
    y = (tf.cast(tf.argmax(y, axis=-1), tf.float32) + 1.0) * N
    y = tf.expand_dims(y, axis=-1)

    # Set logits tensor so that loss function will use it as predicted values
    tfr.context.set_logits_tensor(y, logits)
    return y

