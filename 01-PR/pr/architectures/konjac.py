from tframe import context
from tframe import mu as m
from tframe.layers.hyper.conv import ConvBase
from pr_core import th

import tensorflow as tf


def _fetch_prior() -> tf.Tensor:
  input_layer = context.get_from_pocket(th.PRKeys.prior)
  assert isinstance(input_layer, m.Input)
  if input_layer.place_holder is None: input_layer()
  assert isinstance(input_layer.place_holder, tf.Tensor)
  return input_layer.place_holder


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





