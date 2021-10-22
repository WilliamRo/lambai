from tframe import context
from tframe import mu as m
from tframe.layers.hyper.conv import ConvBase
from tframe.operators.apis.fourier.basis import get_fourier_basis
from pr_core import th

from tframe import tf

import numpy as np


def _fetch_prior() -> tf.Tensor:
  input_layer = context.get_from_pocket(th.PRKeys.prior)
  assert isinstance(input_layer, m.Input)
  if input_layer.place_holder is None: input_layer()
  assert isinstance(input_layer.place_holder, tf.Tensor)
  return input_layer.place_holder


def dual_base(self: ConvBase, shape: tuple):
  # shape = [L, L, 1, C]
  L, C = shape[0], shape[-1]
  assert self.abbreviation == 'duconv2d'

  # Get the package
  # shape = [?, 2, 2], prior[k, 0] is center, prior[k, 1] is unit vector
  prior = _fetch_prior()
  assert len(prior.shape) == 3 and prior.shape.as_list()[1:] == [2, 2]
  #: [?, 2]
  center = prior[:, 0]
  #: [?, 2]
  uv = prior[:, 1]

  # [1/3] theta
  # Create learnable variables: r (N,), theta (N,), radius (N,)/None
  N = int(360 / th.kon_omega)
  theta_init = np.linspace(0., (N - 1) / N, N, dtype=np.float32)
  theta = tf.get_variable('theta', dtype=th.dtype, initializer=theta_init)
  # theta will be multiplied by 2*pi when used to create rotation matrix
  context.add_to_list_collection('dual', theta)

  # [2/3] d
  assert len(th.kon_rs) == 1

  # TODO to be deprecated
  # # Reverse sigmoid
  # y = th.kon_rs[0]
  # x = np.log(y / (1. - y))
  # r_init = np.array([x] * N, dtype=np.float32)
  # r = tf.get_variable('r_from_origin', dtype=th.dtype, initializer=r_init)
  # # r should be in [0, 1]
  # r = tf.sigmoid(r)

  r_init = np.array([th.kon_rs[0] / th.n2o] * N, dtype=np.float32)
  r = tf.get_variable('r_from_origin', dtype=th.dtype, initializer=r_init)
  r = r * th.n2o
  context.add_to_list_collection('dual', r)

  # [3/3] r
  assert isinstance(th.kon_rad, float)
  radius_init = np.array([th.kon_rad / th.n2o] * N, dtype=np.float32)
  radius = tf.get_variable('pupil_rad', dtype=th.dtype, initializer=radius_init)
  # radius = tf.abs(radius)  # not needed maybe
  radius = radius * th.n2o
  context.add_to_list_collection('dual', radius)

  #: [?, L, L, C]
  real, imag = get_fourier_basis(th.prior_size, center=center, uv=uv, r=r,
                                 theta=theta, radius=radius, fmt='c')
  #: [?, L, L, 1, C]
  return tf.expand_dims(real, -2), tf.expand_dims(imag, -2)

  # ---------------------------------------------------------------------------
  # # shape = [K, K, 1, O]
  # K, O = shape[0], shape[-1]
  # # kernel.shape is [?, K, K, O]
  # kernel = _fetch_prior()
  #
  # kernel = tf.reshape(kernel, shape=[-1, K, K, 1, O])
  # assert kernel.shape.as_list()[1:] == list(shape)
  # return kernel


def plot_dual(steps: list, package: list):
  from lambo import DaVinci
  import matplotlib.pyplot as plt

  # box[highlight_index, less_factor]
  box = [0, 1]
  total = len(steps)
  channels = len(package[0][0])
  def _get_plotter(i=None):
    def _plot(ax: plt.Axes):
      assert i is not None

      # Draw unit circle
      ax.add_patch(plt.Circle(
        (0, 0), 1.0, fill=False, color='grey', alpha=0.1))

      # Draw each channel in k-space
      for k, (theta, r, radius) in enumerate(zip(*package[i])):
        if k % box[1] != 0: continue
        theta = 2 * np.pi * theta
        # print(f'r = {r}')
        x, y = r * np.cos(theta), r * np.sin(theta)

        # Plot K-space position
        COLORS = ['red', 'orangered', 'tomato', 'salmon', 'peachpuff',
                  'orange', 'gold', 'yellow', 'greenyellow', 'chartreuse',
                  'palegreen', 'lime', 'springgreen', 'aquamarine',
                  'turquoise', 'cyan', 'lightskyblue', 'deepskyblue',
                  'dodgerblue', 'blue', 'slateblue', 'blueviolet', 'violet',
                  'purple', 'deeppink', 'hotpink']
        color = COLORS[k % len(COLORS)]
        ax.plot(x, y, '.', color=color, linewidth='1')

        # Visualize pupil size
        R = 0.1
        alpha = 0.5
        highlight = k == box[0]
        ax.add_patch(plt.Circle(
          (x, y), R * radius, fill=highlight, color=color, alpha=alpha))

      # Finalize
      MAX = 1.1
      ax.set_xlim(-MAX, MAX), ax.set_ylim(-MAX, MAX)
      ax.grid(True), ax.set_axis_off()
      title = f'Epoch {steps[i]}/{steps[-1]}'
      K = box[0]
      if 0 <= K < len(package[0][0]):
        theta, r, radius = [lst[K] for lst in package[i]]
        title += f', channel[{K}]: $\\theta$ = {theta * 360:.1f}'
        title += f', r = {r:.3f}, pupil-r = {radius:.3f}'
      ax.set_title(title)
    return _plot

  da = DaVinci('Pupil Viewer', 7, 7)

  # Define functions
  def lock(d: int = 1):
    assert d in (-1, 1)
    box[0] += d * box[1]
    _total = channels // box[1] * box[1]
    if box[0] < 0: box[0] = _total
    if box[0] > _total: box[0] = 0
    da.refresh()
  def less(k: int = 1):
    box[1] = k
    box[0] = 0   # reset highlight index
    da.refresh()
  da.less = less
  da.state_machine.register_key_event('n', lambda: lock(1))
  da.state_machine.register_key_event('p', lambda: lock(-1))

  for i in range(total): da.add_plotter(_get_plotter(i))
  da.show()


def cube_solver(self: ConvBase, shape: tuple) -> tf.Tensor:
  # The filter produced by this method is used for convolving images with
  # .. arbitrary channels, shape = [K, K, in_shape:M, out_shape:=N]
  assert len(shape) == 4 and shape[0] == shape[1]
  K, N, M, S = shape[0], shape[-1], shape[2], th.prior_size
  # assert M == 1

  # Fetch prior tensor with shape [?, S, S, 2]
  prior = _fetch_prior()
  # prior = tf.expand_dims(prior, axis=-1)
  assert len(prior.shape) == 4
  assert S % K == 0
  factor = S // K

  # Kernel shape will be [?, K, K, N*M]
  kernel = self.conv2d(prior, N * M, filter_size=factor,
                       scope='hyper_conv', strides=factor)

  # Activate if necessary
  if th.kon_activation is not None:
    kernel = m.Activation(th.kon_activation)(kernel)
    if th.kon_activation == 'sigmoid':
      kernel = kernel - 0.5

  # Expand kernel dimension to [None, K, K, M, N] and return
  return tf.reshape(kernel, shape=[-1, K, K, M, N])


def dettol(self: ConvBase, shape: tuple) -> tf.Tensor:
  # shape = [K, K, in_shape:=M, out_shape:=N]
  # kernel size K should be at least 9
  assert len(shape) == 4 and shape[0] == shape[1]
  K, N, M, S = shape[0], shape[-1], shape[2], th.prior_size

  # Fetch prior tensor with shape [?, S, S, 2]
  prior = _fetch_prior()
  assert len(prior.shape) == 4
  assert K == S

  # Kernel shape will be [?, K, K, N*M]
  kernel = self.dense(M*N, prior, 'hyper_kernel')
  assert kernel.shape.as_list() == [None, K, K, N*M]

  # Expand kernel dimension to [None, K, K, M, N] and return
  return tf.reshape(kernel, shape=[-1, K, K, M, N])








