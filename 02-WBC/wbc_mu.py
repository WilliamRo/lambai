from tframe import Classifier
from tframe.nets.net import Net
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

from wbc_arch.rethinker import Rethinker
from wbc.wbc_hub import WBCHub


def get_container(th, flatten=False, add_last_dim=True):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean))
  if add_last_dim:
    model.add(Reshape(shape=th.input_shape + [1]))
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


def add_rethink(th, model, fc_dims, index_group=(0, 1)):
  assert isinstance(th, WBCHub) and isinstance(model, Classifier)
  if th.use_wise_man:
    rth = Rethinker(index_group, loss_coef=th.loss_coef)
    fm = model.add_forkmerge(
      rth, name='WiseMan', stop_gradient_at=['branch'] if th.stop_grad else [])

  for branch_key in ('main', 'branch') if th.use_wise_man else ('main',):
    add = ((lambda l: fm.add_to_branch(branch_key, l))
         if th.use_wise_man else model.add)
    for dim in fc_dims:
      if th.dropout > 0: add(Dropout(1. - th.dropout))
      add(Dense(dim, activation=th.activation))
    # Add softmax units
    num = th.num_classes if branch_key == 'main' else len(index_group)
    add(Dense(num))
    # Only add softmax to main branch
    if branch_key == 'main': add(Activation('softmax'))


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


