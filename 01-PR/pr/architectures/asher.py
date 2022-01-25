from typing import Optional

import numpy as np

from tframe import tf
from tframe import mu as m
from tframe.nets.classic.conv_nets.conv_net import ConvNet


class Asher(ConvNet):

  def __init__(self, auto_mark=True):
    from pr_core import th

    # Call parent's constructor
    super(Asher, self).__init__()

    # Append structure tail to mark if required
    if auto_mark: th.mark += '({})'.format(self.mark_tail)

  # region: Properties

  @property
  def mark_tail(self):
    from pr_core import th
    tail = f'f{th.filters}-ks{th.kernel_size}-di{th.dilations}'
    tail += f'-{th.ash_token}'
    return f'{tail}'

  # endregion: Properties

  # region: Sub-structures

  @staticmethod
  def conv(ch, ks=1, di=1, ac: Optional[str] = None, name=None):
    from pr_core import th
    return m.HyperConv2D(filters=ch, kernel_size=ks, dilations=di,
                         activation=ac, use_bias=th.use_bias, name=name)

  def _multi_dilation(self, block_token: str, kernel_size: int, configs: list):
    """configs is a list of tuple (dilation, channel), e.g.,
          [(1, 32), (2, 10), (4, 10), (8, 10)]
    """
    from pr_core import th

    # Initialize layers
    layers = []

    # Construct vertices
    vertices = [self.conv(dc[1], kernel_size, dc[0], ac=th.activation)
                for dc in configs]

    if len(vertices) == 1: return vertices

    # Construct a DAG
    edges = ';'.join(['1' + '0' * i for i in range(len(vertices))])
    # Add merge layer
    vertices.append(m.Merge.Concat())
    edges += ';0' + '1' * (len(vertices) - 1)
    layers.append(m.ForkMergeDAG(vertices, edges, name=f'Ash-{block_token}'))

    return layers

  # endregion: Sub-structures

  # region: Implementation

  def _get_layers(self):
    from pr_core import th

    # Sanity check
    assert th.thickness > 0

    # Init layers
    layers = []

    filters = th.filters
    knl_size = th.kernel_size

    while filters > 0:
      ks_int = int(knl_size)
      # Last layer
      if filters == 1:
        layers.append(m.Conv2D(
          1, ks_int, use_bias=False, activation='sigmoid'))
        break

      # Add block
      if th.ash_token == 'alpha':
        configs = [(1, filters)]
      else: raise KeyError

      block_token = f'f{filters}k{ks_int}'
      block = self._multi_dilation(block_token, ks_int, configs)
      layers.extend(block)

      for i in range(th.thickness - 1): layers.append(self.conv(
          filters, ks_int, ac=th.activation, name=f'{block_token}-{i+2}'))

      # Halve channel number
      filters = filters // 2
      knl_size = knl_size * np.sqrt(2)

    return layers

  # endregion: Implementation
