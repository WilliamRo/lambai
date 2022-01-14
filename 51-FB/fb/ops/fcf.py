import numpy as np

from tframe import tf
from tframe.layers.layer import Layer
from tframe.operators.apis.neurobase import NeuroBase

from fb.ops.yolo_helper import YOLOImage
from fb.ops.yolout import YOLOut
from talos.tasks.detection.box import Box

from tframe.nets.classic.conv_nets.conv_net import ConvNet



class FCFinder(Layer, NeuroBase):
  full_name = 'null'
  abbreviation = 'null'

  is_nucleus = True


  @classmethod
  def get_model_detail(cls):
    from fb_core import th
    return f'f{th.filters}h{th.floor_height}' + (
      '-bn' if th.use_batchnorm else '') + (
      '-ab' if th.auto_bound else '')


  @staticmethod
  def add_conv_layers(model):
    from tframe import tf
    from tframe import Predictor
    from tframe.layers.common import Reshape
    from tframe.layers.convolutional import Conv2D
    from tframe.layers.pooling import MaxPool2D, AveragePooling2D
    from fb_core import th

    assert isinstance(model, Predictor)

    def _build_floor(M, n_channels):
      for _ in range(M): model.add(ConvNet.conv_bn_relu(
        n_channels, th.kernel_size, th.use_batchnorm)[0])

    # Input should be halved (using pooling stuff) for N times
    N = np.log2(th.fb_img_size / th.yolo_S)
    # Sanity check
    assert int(N) == N

    n_channels = th.filters
    # Add first conv-layer
    model.add(ConvNet.conv_bn_relu(
      n_channels, th.kernel_size, use_batchnorm=False, expand_last_dim=True)[0])
    _build_floor(th.floor_height - 1, n_channels)

    # Build each floor
    for i in range(int(N)):
      model.add(MaxPool2D(th.kernel_size, strides=2))
      n_channels *= 2
      _build_floor(th.floor_height, n_channels)

    # Add last layer
    model.add(Conv2D(th.yolo_B * 5, kernel_size=1, use_bias=False))
    model.add(Reshape(shape=[th.yolo_S, th.yolo_S, th.yolo_B, 5]))

    if not th.auto_bound: model.add(YOLOut())




