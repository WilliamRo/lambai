from tframe import Classifier
from tframe.layers import Input, Activation, Flatten
from tframe.configs.config_base import Config
from tframe.layers.preprocess import Normalize

from tframe.layers.convolutional import Conv2D
from tframe.layers.pooling import MaxPool2D
from tframe.layers.advanced import Dense

from tframe.layers.common import Dropout
from tframe.layers.common import Reshape


def get_container(th, flatten=False, add_last_dim=True):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  if add_last_dim:
    model.add(Reshape(shape=th.input_shape + [1]))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean))
  if flatten: model.add(Flatten())
  return model


def finalize(th, model, add_output_layer=True):
  assert isinstance(th, Config) and isinstance(model, Classifier)
  # Add output layer
  if add_output_layer:
    model.add(Dense(num_neurons=th.num_classes))
    model.add(Activation('softmax'))
  model.build(th.get_optimizer(), metric=['accuracy', 'loss'],
              batch_metric='accuracy', eval_metric='accuracy')
  return model


def typical(th, layers, flatten=False):
  assert isinstance(th, Config)
  model = get_container(th, flatten)
  if not isinstance(layers, (list, tuple)): layers = [layers]
  for layer in layers: model.add(layer)
  return finalize(th, model)

