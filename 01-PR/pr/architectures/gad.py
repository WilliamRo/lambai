from collections import OrderedDict
from typing import Optional

from tframe import hub as th
from tframe import tf
from tframe import mu as m
from tframe.nets.classic.conv_nets.conv_net import ConvNet


class Gad(ConvNet):

  def __init__(self, auto_mark=True):
    # Call parent's constructor
    super(Gad, self).__init__()

    # Append structure tail to mark if required
    if auto_mark: th.mark += '({})'.format(self.mark_tail)

  # region: Properties

  @property
  def mark_tail(self):
    from pr_core import th

    tail = f'{th.archi_string}-ks{th.kernel_size}-gd{th.dilations}'
    if th.bottle_neck: tail += '-bnk'
    return f'{tail}'

  # endregion: Properties

  # region: Sub-structures

  def _get_gad(self, block_id, channels, kernel_size, projection=False):
    """Get a gad block implementing a highway operator using Conv2D:

    input x (shape: [?, H, W, C])
      x_bar = phi(Conv2D(x))                     # candidate
       gate = sigmoid(Conv2D(x))                 # input gate
          y = gate * x_bar + (1 - gate) * x      # highway output
    """
    from pr_core import th

    def conv(ch, ks=1, di=1, ac: Optional[str] = None, name = None):
      return m.HyperConv2D(filters=ch, kernel_size=ks, dilations=di,
                           activation=ac, use_bias=th.use_bias, name=name)

    # Initialize branches
    x_bar, gate = [], []
    # Add bottle neck without activation if required
    if th.bottle_neck and block_id > 0:
      for branch in (x_bar, gate): branch.append(conv(channels // 2))

    # Append main layers to branches
    x_bar_ac = None if th.activation == '-' else th.activation
    x_bar.append(conv(channels, kernel_size, ac=x_bar_ac, name='x_bar'))
    gate.append(conv(channels, kernel_size, ac='sigmoid', di=th.dilations,
                     name=f'gate-di{th.dilations}'))

    # Create and return DAG
    vertices, edges = [x_bar, gate, m.Merge.Highway()], '1;10;111'
    if projection:
      vertices.insert(0, conv(channels, name='projection'))
      edges = '1;10;100;0111'
    return m.ForkMergeDAG(vertices, edges, name=f'Gad-{block_id + 1}')

  # endregion: Sub-structures

  # region: Implementation

  def _get_layers(self):
    channels = [int(n) for n in th.archi_string.split('-')]
    layers = []

    for i, c in enumerate(channels):
      projection = i == 0 or c != channels[i - 1]
      layers.append(self._get_gad(i, c, th.kernel_size, projection))

    return layers

  # endregion: Implementation
