from collections import OrderedDict
from typing import Optional

import numpy as np

from tframe import tf
from tframe import mu as m
from tframe.nets.classic.conv_nets.conv_net import ConvNet


class Naphtali(ConvNet):

  def __init__(self, auto_mark=True):
    from pr_core import th

    # Call parent's constructor
    super(Naphtali, self).__init__()

    # Append structure tail to mark if required
    if auto_mark: th.mark += '({})'.format(self.mark_tail)

  # region: Properties

  @property
  def mark_tail(self):
    from pr_core import th
    tail = f'{th.archi_string}-ks{th.kernel_size}-di{th.dilations}'
    tail += f'-{th.nap_token}'
    # tail += f'-la({th.local_activation})-ga({th.global_activation})'
    # tail += f'-{th.nap_merge}'
    # if th.input_projection: tail += '-proj'
    return f'{tail}'

  # endregion: Properties

  # region: Sub-structures

  @staticmethod
  def conv(ch, ks=1, di=1, ac: Optional[str] = None):
    from pr_core import th
    return m.HyperConv2D(filters=ch, kernel_size=ks, dilations=di,
                         activation=ac, use_bias=th.use_bias)

  def _multi_dilation(self, block_id: int, kernel_size: int, configs: list):
    """configs is a list of tuple (dilation, channel), e.g.,
          [(1, 32), (2, 10), (4, 10), (8, 10)]
    """
    from pr_core import th

    out_channels = sum([dc[1] for dc in configs]) // 2

    # Initialize layers, add projection before non-linear operations calculation
    layers = []
    if block_id > 0 and th.input_projection:
      layers.append(self.conv(out_channels))

    # Construct vertices
    vertices = [self.conv(dc[1], kernel_size, dc[0], ac=th.activation)
                for dc in configs]

    # Construct a DAG
    edges = ';'.join(['1' + '0' * i for i in range(len(vertices))])
    # Add merge layer
    vertices.append(m.Merge.Concat())
    edges += ';0' + '1' * (len(vertices) - 1)
    layers.append(m.ForkMergeDAG(vertices, edges, name=f'Nap-{block_id + 1}'))

    return layers

  def _block_alpha(
      self, block_id, filters: int, kernel_size: int, global_dilations: int):
    """Potential adjustments:
       (1) put input projection at the end
    """
    from pr_core import th

    # Initialize branches
    branch_local, branch_global = [], []
    if th.input_projection and block_id > 0:
      for branch in (branch_local, branch_global):
        branch.append(self.conv(filters))

    # Append main layers to branches
    la = None if th.local_activation == '-' else th.local_activation
    ga = None if th.global_activation == '-' else th.global_activation
    branch_local.append(self.conv(filters, kernel_size, ac=la))
    branch_global.append(self.conv(
      filters, kernel_size, di=global_dilations, ac=ga))

    # Define merge layer
    assert th.nap_merge in ('concat', 'cross-concat')
    merge_layer = (m.Merge.Concat() if th.nap_merge == 'concat'
                   else m.Merge.CrossConcat())

    # Create and return DAG
    return m.ForkMergeDAG(
      vertices=[branch_local, branch_global, merge_layer],
      edges='1;10;011', name=f'Nap-{block_id + 1}')

  def _block_beta(self, block_id, channels, kernel_size, dilations: int):
    """Create a uniformly distributed dilation setup.
       Some times the 3rd parameter is `filters` and sometimes it's `channels`
    """
    from pr_core import th

    # Add projection before non-linear operations calculation
    layers = []
    if block_id > 0 and th.input_projection: layers.append(self.conv(channels))

    # .
    vertices = []

    # Construct local branch
    vertices.append(self.conv(channels, kernel_size, ac=th.activation))

    # Construct global branch
    order = int(np.log2(dilations))
    c = channels // order
    for o in range(order): vertices.append(
      self.conv(c, kernel_size, di=2 ** (o + 1), ac=th.activation))

    # Construct a DAG
    vertices.append(m.Merge.Concat())
    edges = ';'.join(['1' + '0' * i for i in range(order + 1)])
    edges += ';0' + '1' * (len(vertices) - 1)
    layers.append(m.ForkMergeDAG(vertices, edges, name=f'Nap-{block_id + 1}'))

    # Return layers
    return layers

  def _block_gamma(self, block_id, channels, kernel_size, dilations: int):
    from pr_core import th

    # Add projection before non-linear operations calculation
    layers = []
    if block_id > 0 and th.input_projection: layers.append(self.conv(channels))

    # .
    vertices = []

    # Construct  branches
    if channels < dilations: channels = dilations
    c = channels // dilations
    for d in range(1, dilations + 1): vertices.append(
      self.conv(c, kernel_size, di=d, ac=th.activation))

    # Construct a DAG
    vertices.append(m.Merge.Concat())
    edges = ';'.join(['1' + '0' * i for i in range(dilations)])
    edges += ';0' + '1' * (len(vertices) - 1)
    layers.append(m.ForkMergeDAG(vertices, edges, name=f'Nap-{block_id + 1}'))

    # Return layers
    return layers

  # endregion: Sub-structures

  # region: Implementation

  def _get_layers(self):
    from pr_core import th
    layers = []

    for i, n in enumerate(th.archi_string.split('-')):
      # Convert n from string to integer
      n = int(n)

      # Get block
      configs = [(1, n)]
      if th.nap_token == 'alpha': configs.append((th.dilations, n))
      elif th.nap_token == 'beta':
        order = int(np.log2(th.dilations))
        for o in range(1, order + 1): configs.append((2 ** o, n // order))
      # elif th.nap_token == 'gamma':
      #  pass
      else: raise KeyError

      block = self._multi_dilation(i, th.kernel_size, configs)

      # Append or extend layer list
      if isinstance(block, list): layers.extend(block)
      else: layers.append(block)

    return layers

  # endregion: Implementation
