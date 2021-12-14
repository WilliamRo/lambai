from tframe import tf
from tframe.layers.layer import Layer
from tframe.operators.apis.neurobase import NeuroBase



class DoNothing(Layer, NeuroBase):
  full_name = 'do-nothing'
  abbreviation = 'dno'

  is_nucleus = False

  def __init__(self, units):
    # Call parent's constructor
    super(DoNothing, self).__init__()
    self.units = units


  @property
  def structure_tail(self):
    return f'(units-{self.units})'


  def _link(self, x: tf.Tensor):
    return x