from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe import mu as m
from tframe.layers.hyper.conv import Conv2D, Deconv2D
from tframe.layers.pooling import MaxPool2D
from tframe.layers.merge import Bridge, Merge, ShortCut
from tframe import activations
from tframe.layers.normalization import BatchNormalization
from tframe.nets.classic.conv_nets.conv_net import ConvNet

from typing import List, Optional, Union


class Ye_UNet2D(ConvNet):

  def __init__(self, filters: Optional[int] = None,
               kernel_size: Optional[int] = 3,
               activation: str = 'relu',
               height: Optional[int] = 4,
               use_batchnorm: bool = False,
               link_indices: Union[List[int], str, None] = 'a',
               auto_crop=False, arc_string: Optional[str] = None):
    """This class provides some generalization to the traditional U-Net

    Example U-Net, height=3, thickness=2, link_indices=2,3

           Left Tower                                            Right Tower
                               bridge index 3
    3F  0   |->|->| --------------------------------------------> |->|->|
                  \                                              /
               contract                                      expand
                   \              bridge index 2              /
    2F  1          |->|->| --------------------------> |->|->|
                         \                            /
                      contract                    expand
                          \                        /
    1F  2                 |->|->|           |->|->|
                                \          /
                             contract  expand
                                 \      /
    GF                           |->|->|         # thickness=2 means 2 `->`s


    :param filters: initial filters, will be doubled after contracting, and
                    halved after expanding
    :param kernel_size: kernel size for each [De]Conv2D layer
    :param activation: activation used in each [Dd]Conv2D layer
    :param height: height of each tower
    :param thickness: number of convolutional layers used on each floor
    :param use_maxpool: whether to use MaxPool2D layer for contracting
    :param use_batchnorm: whether to use BatchNorm layer before activation
    :param link_indices: specifies the floor number to build bridge between
                         2 towers
    :param arc_string: architecture string, if provided, some of the arguments
                       will be overwrote
    """

    self.filters = filters
    self.kernel_size = kernel_size
    self.activation = activation
    self.height = height
    self.use_batchnorm = use_batchnorm
    self.link_indices = link_indices
    self.arc_string = arc_string
    self.auto_crop = auto_crop

    self.parse_arc_str_and_check()


    # TODO: not working and there's no solution
    assert not auto_crop


  def _get_conv(self, filters, kernel_size, strides=1,transpose=False,
                activation='lrelu:0.2', use_batchnorm=True):
    Conv = Deconv2D if transpose else Conv2D
    return Conv(
      filters, kernel_size, strides, padding='same', activation=activation,
      use_bias=True, use_batchnorm=use_batchnorm)

  def _get_layers(self):
    layers, floors = [], []

    # Define some utilities
    contract = lambda channels: m.ForkMergeDAG(
      vertices = [
        m.Conv2D(channels, self.kernel_size, strides=2,
                 activation='lrelu:0.2', use_batchnorm = self.use_batchnorm),
        m.Conv2D(channels, self.kernel_size, strides=2, activation=None,
                 use_batchnorm = self.use_batchnorm),
        m.Merge.Sum(),
        m.Activation(self.activation)],
      edges='1;10;011;0001')

    expand = lambda channels: m.ForkMergeDAG(
      vertices = [
        m.Deconv2D(channels, self.kernel_size, strides=2,
                   activation=self.activation,
                   use_batchnorm=self.use_batchnorm)],
      edges = '1')

    # Build left tower for contracting
    filters = self.filters

    layers.append(m.Conv2D(self.filters, self.kernel_size, strides=1,
                           activation=self.activation, use_batchnorm=True))

    for i in range(self.height):   # (height - i)-th floor
      # Add front layers on each floor
      layers.append(m.ForkMergeDAG(
        vertices=[
          m.Conv2D(filters, self.kernel_size, strides=1,
                   activation=self.activation, use_batchnorm=True),
          m.Conv2D(filters, self.kernel_size, strides=1, activation=None,
                   use_batchnorm = True),
          m.Merge.Sum(), m.Activation(self.activation)],
        edges='1;01;101;0001'))

      # Remember the last layer in each floor before contracting
      floors.append(layers[-1])
      # Double_filters
      filters *= 2
      # Contract
      if i < self.height-1:
        layers.append(contract(filters))

    # Build ground floor (GF)
    layers.append(m.ForkMergeDAG(
      vertices=[
        m.Conv2D(filters, self.kernel_size, strides=2,
                 activation=self.activation, use_batchnorm=True),
        m.Conv2D(filters, self.kernel_size, strides=1,
                 activation=self.activation, use_batchnorm=True),
        m.Conv2D(filters, self.kernel_size, strides=1, activation=None,
                 use_batchnorm = True),
        m.Merge.Sum(), m.Activation(self.activation)],
      edges='1;01;001;0101;00001'))

    # Build right tower for expanding
    for i in range(1, self.height + 1):    # i-th floor
      # Halve filters
      filters = filters // 2
      # Expand
      layers.append(expand(filters))
      # Build a bridge if necessary
      if i in self.link_indices:
        '''guest_is_larger = None
        if self.auto_crop: guest_is_larger = not self.use_maxpool'''
        layers.append(Bridge(floors[self.height - i], guest_first=False))

      layers.append(m.Conv2D(filters, kernel_size=1, strides=1,
                             use_batchnorm=True, activation=self.activation))

      layers.append(m.ForkMergeDAG(
        vertices=[m.Conv2D(filters, self.kernel_size, strides=1,
                           activation=self.activation, use_batchnorm=True),
                  m.Conv2D(filters, self.kernel_size, strides=1,
                           activation=None, use_batchnorm=True),
                  m.Merge.Sum(), m.Activation(self.activation)],
        edges='1;01;101;0001'))

    return layers


  def parse_arc_str_and_check(self):
    """The format of arc_string is
      {filters}-{kernel_size}-{height}-{thickness}-[link_indices]-[mp]-[bn]
    in which
      {link_indices} can be `a` or `f` indicating linking all layers on the same
      floor, or indices separated by `,` indicating which floor to link, e.g.,
      `0,2,4`, in which case the given height must be greater than 4.

    Note that arc_string does not support different kernel size for left and
      right tower.
    """
    if self.arc_string is not None:
      options = self.arc_string.split('-')
      assert len(options) >= 4
      self.filters, self.kernel_size, self.height = [int(op) for op in options[:3]]
      self.activation = options[3]
      # For optional settings
      for op in options[4:]:
        if op in ('bn', 'batchnorm'): self.use_batchnorm = True
        elif op in ('a', 'f', 'all', 'full'): self.link_indices = op
        else:
          # Parse link_indices with weak format checking
          assert ','in op and self.link_indices is None
          self.link_indices = [int(ind) for ind in op.split(',')]
          assert len(self.link_indices) == len(set(self.link_indices))

    # Check types
    checker.check_positive_integer(self.height)
    checker.check_positive_integer(self.filters)
    checker.check_type(self.activation, str)
    checker.check_positive_integer(self.kernel_size)
    checker.check_type(self.use_batchnorm, bool)
    if self.link_indices in (None, 'none', '-', ''):
      self.link_indices = []
    elif self.link_indices in ('a', 'f', 'all', 'full'):
      self.link_indices = list(range(1, self.height + 1))
    if self.link_indices: checker.check_type(self.link_indices, int)


  def __str__(self):
    result = '{}-{}-{}-{}-{}'.format(
      self.height, self.activation)
    if len(self.link_indices) < self.height:
      if len(self.link_indices) == 0: result += '-o'
      else: result += '-' + ','.join([str(i) for i in self.link_indices])
    if self.use_batchnorm: result += '-bn'
    return result


if __name__ == '__main__':
  model = m.Predictor('Ye')
  model.add(m.Input(sample_shape=(512,512,1)))
  Ye_UNet2D(filters=16, kernel_size=3, activation='lrelu:0.2', height=4,
         use_batchnorm=True).add_to(model)
  model.rehearse(export_graph=True)
  print()


