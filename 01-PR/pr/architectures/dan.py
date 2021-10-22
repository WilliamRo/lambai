from collections import OrderedDict

from tframe import hub as th
from tframe import tf
from tframe import mu as m
from tframe.nets.classic.conv_nets.conv_net import ConvNet


class Dan(ConvNet):

  def __init__(self, auto_mark=True):
    # Call parent's constructor
    super(Dan, self).__init__()

    # Append structure tail to mark if required
    if auto_mark: th.mark += '({})'.format(self.mark_tail)

    # m.DUC()

  # region: Properties

  @property
  def mark_tail(self):
    tail = 'b{}'.format(th.num_blocks)
    tail += '-f{}'.format(th.filters)
    tail += '-k{}'.format(th.kernel_size)
    if th.use_batchnorm: tail += '-bn'
    return tail

  # endregion: Properties

  # region: Sub-structures

  def _block_alpha(self, block_id, filters: int = None, batchnorm: bool = None,
                   dilations: int = None):
    """block_id begins from 0"""
    if filters is None: filters = th.filters
    if batchnorm is None: batchnorm = th.use_batchnorm
    if dilations is None: dilations = th.dilations

    assert filters % 2 == 0

    conv = (lambda m=1, k=th.kernel_size, s=1, t=False, d=dilations:
            self.conv_bn_relu(filters // 2 * m, k, batchnorm,
                              th.relu_leak, s, t, d)[0])

    return [m.ForkMergeDAG(
      vertices=[conv(k=1, m=2),
                [conv(k=1), conv()],
                [conv(k=1), conv(m=2, s=2), conv(s=2, t=True, d=1)],
                m.Merge.ConcatSum()],
      edges='1;10;100;0111',
      name='Block_{}'.format(block_id + 1), auto_merge=True)]

  # endregion: Sub-structures

  # region: Implementation

  def _get_layers(self):
    layers = []

    for i in range(th.num_blocks):
      layers.extend(self._block_alpha(i))

    return layers

  # endregion: Implementation
