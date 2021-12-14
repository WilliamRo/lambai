from tframe import tf
from tframe.layers.layer import Layer
from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.common import Input
from tframe.layers.normalization import BatchNormalization



class Joint(Layer, NeuroBase):
  full_name = 'joint'
  abbreviation = 'joint'

  is_nucleus = True


  @property
  def structure_tail(self):
    from pet_core import th

    if not th.use_meta_data: return ''
    if th.by_pass_image: return f'(m->{th.meta_branch_code})'
    return f'(i+m->{th.meta_branch_code})'


  def _link(self, x: tf.Tensor):
    from pet_core import th

    if not th.use_meta_data: return x

    # Create a placeholder for metadata
    from pet.pet_set import PetSet

    u = Input([12], name=PetSet.META)()

    for i, n in enumerate(th.meta_branch_code.split('-')):
      with tf.variable_scope(f'Meta_{i+1}'):
        n = int(n)
        assert n > 0
        u = self.dense_v2(n, f'meta_{i+1}', u, activation=th.activation)
        if th.use_batchnorm: u = BatchNormalization()(u)

    if th.by_pass_image: return u
    return tf.concat([x, u], axis=1, name='im_meta')



