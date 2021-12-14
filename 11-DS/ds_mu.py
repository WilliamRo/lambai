from tframe import tf

from tframe import Predictor
from tframe.nets.classic.conv_nets.lenet import LeNet

from tframe.layers import Input, Linear, Activation, Flatten
from tframe.layers.normalization import BatchNormalization
from tframe.layers.preprocess import Normalize

from tframe.layers.common import Dropout
from tframe.layers.normalization import BatchNormalization, LayerNormalization
from tframe.layers.convolutional import Conv2D
from tframe.layers.pooling import MaxPool2D, AveragePooling2D
from tframe.layers.pooling import GlobalAveragePooling2D
from tframe.layers.highway import LinearHighway
from tframe.layers.merge import ShortCut
from tframe.layers.hyper.dense import Dense

from pet.ops.pet_out import PetOut
from pet.ops.joint import Joint
from pet.ops.do_nothing import DoNothing

from pet_core import th


def get_container():
  model = Predictor(mark=th.mark)
  model.add(Input(sample_shape=th.pet_input_shape))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean, sigma=255.))
  return model


def finalize(model, add_output_layer=True):
  assert isinstance(model, Predictor)

  model.add(DoNothing(units=10))

  if th.use_meta_data: model.add(Joint())

  # Add output layer if required
  if add_output_layer: model.add(PetOut())

  # Build model
  if th.use_classifier: th.target_dim = th.num_classes
  loss = 'mse' if not th.use_classifier else 'cross_entropy'
  model.build(loss=loss, metric='rmse', batch_metric='rmse')
  return model


def conv_bn_act(filters, strides=1):
  return Conv2D(filters, strides=strides, activation=th.activation,
                kernel_size=th.kernel_size, use_batchnorm=th.use_batchnorm)


def construct_alpha(model):
  assert isinstance(model, Predictor)
  for i, c in enumerate(th.archi_string.split('-')):
    c = int(c)
    # Shrink if necessary
    if i > 0:
      model.add(conv_bn_act(c, strides=2))
      if th.dropout > 0: model.add(Dropout(th.dropout))
    model.add(conv_bn_act(c))

  # Add a global average pooling layer
  model.add(GlobalAveragePooling2D())



