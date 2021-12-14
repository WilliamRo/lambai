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

from fb_core import th



def get_container():
  model = Predictor(mark=th.mark)
  model.add(Input(sample_shape=[th.fb_img_size, th.fb_img_size]))
  return model


def finalize(model):
  assert isinstance(model, Predictor)

  # Build model
  model.build(loss='mse', metric='rmse', batch_metric='rmse')
  return model


def add_feature_extractor(model, channels=8, n_blocks=3, flatten=False):
  assert isinstance(model, Predictor)
  # 100x100x1
  model.add(Conv2D(channels, kernel_size=3, activation='relu',
                   expand_last_dim=True))
  for _ in range(n_blocks):
    channels *= 2
    model.add(Conv2D(channels, kernel_size=th.kernel_size,
                     strides=2, activation='relu'))
    model.add(Conv2D(channels, kernel_size=3, activation='relu'))

  if flatten: model.add(Flatten())
