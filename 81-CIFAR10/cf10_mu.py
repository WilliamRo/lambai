import tensorflow as tf

from tframe import Classifier
from tframe.layers import Input, Linear, Activation, Flatten
from tframe.layers.normalization import BatchNormalization
from tframe.layers.preprocess import Normalize

from tframe.layers.common import Dropout
from tframe.layers.normalization import BatchNormalization
from tframe.layers.convolutional import Conv2D
from tframe.layers.pooling import MaxPool2D, AveragePooling2D
from tframe.layers.highway import LinearHighway
from tframe.layers.merge import ShortCut
from tframe.layers.hyper.dense import Dense

from cf10_core import th


def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean, sigma=255.))
  if flatten: model.add(Flatten())
  return model


def finalize(model, flatten=False, add_output_layers=True):
  assert isinstance(model, Classifier)
  if flatten: model.add(Flatten())
  # Add output layer
  if add_output_layers:
    model.add(Dense(num_neurons=th.num_classes))
    model.add(Activation('softmax'))
  # Build model
  model.build(metric=['accuracy', 'loss'], batch_metric='accuracy',
              eval_metric='accuracy')
  return model


