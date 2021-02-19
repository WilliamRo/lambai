from tframe import Classifier
from tframe.layers import Input, Activation, Flatten
from tframe.configs.config_base import Config
from tframe.layers.preprocess import Normalize

from tframe.layers.convolutional import Conv2D
from tframe.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from tframe.layers.advanced import Dense

from tframe.layers.normalization import BatchNormalization
from tframe.layers.normalization import LayerNormalization
from tframe.layers.common import Activation
from tframe.layers.common import Dropout
from tframe.layers.common import Reshape

from tframe.layers.merge import ShortCut


def get_container(th, flatten=False, add_last_dim=True):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  if add_last_dim:
    model.add(Reshape(shape=th.input_shape + [1]))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean))
  if flatten: model.add(Flatten())
  return model


def finalize(th, model, add_output_layer=True, flatten=False):
  assert isinstance(th, Config) and isinstance(model, Classifier)
  if flatten: model.add(Flatten())
  # Add output layer
  if add_output_layer:
    model.add(Dense(num_neurons=th.num_classes))
    model.add(Activation('softmax'))
  # Determine metric
  assert th.early_stop_metric in ('f1', 'accuracy')
  metric = ['f1', 'accuracy'] if th.early_stop_metric == 'f1' else [
    'accuracy', 'f1']
  # Build model
  model.build(th.get_optimizer(), metric=metric,
              batch_metric='accuracy', eval_metric=th.eval_metric)
  return model


def typical(th, layers, flatten=False):
  assert isinstance(th, Config)
  model = get_container(th, flatten)
  if not isinstance(layers, (list, tuple)): layers = [layers]
  for layer in layers: model.add(layer)
  return finalize(th, model)


# region: Converted from Xin's codes

def add_bn_relu(model):
  assert isinstance(model, Classifier)
  model.add(BatchNormalization())
  model.add(Activation.ReLU())


def add_basic_block(
    model, filters, kernel_size=3, init_strides=1, is_first_unit=False):
  assert isinstance(model, Classifier)
  h = model.last_function
  if not is_first_unit: add_bn_relu(model)
  model.add(Conv2D(filters, kernel_size, init_strides))
  # Second part of [bn-relu-]conv
  add_bn_relu(model)
  model.add(Conv2D(filters, kernel_size, strides=1))
  # Add shortcut
  transforms = None if init_strides == 1 else [
    [Conv2D(filters, kernel_size=1, strides=2)]]
  model.add(ShortCut(h, mode=ShortCut.Mode.SUM, transforms=transforms))


def add_residual_unit(
    model, filters, repetitions, unit_id, kernel_size=3):
  assert isinstance(model, Classifier)
  is_first_unit = unit_id == 0
  for i in range(repetitions):
    init_strides = 2 if i == 0 and unit_id > 0 else 1
    add_basic_block(
      model, filters, kernel_size, init_strides, is_first_unit)

# endregion: Converted from Xin's codes


