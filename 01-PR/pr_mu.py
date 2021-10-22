import tensorflow as tf

from tframe import context
from tframe import Predictor
from tframe import losses
from tframe import mu as m
from pr_core import th

import pr.architectures.konjac as konjac


def get_hyper_channels():
  if th.prior_key == 'dual':
    return int(len(th.kon_rs) * 360 / th.kon_omega)

  # TODO: weed out these old routines after finding best hyper
  coef = (2 if th.prior_format in ('dc', 'doulbe_channel', 'default')
          else 1)
  return int(len(th.kon_rs) * 180 / th.kon_omega) * coef


def get_container():
  model = m.Predictor(mark=th.mark)
  model.add(m.Input(sample_shape=th.input_shape))

  if th.use_prior:
    assert isinstance(th.prior_size, int) and th.prior_size > 0
    shape = [th.prior_size, th.prior_size, 2]
    if th.prior_key in ('dettol', ):
      shape[-1] = get_hyper_channels()
    elif th.prior_key == 'dual':
      shape = [2, 2]
    context.put_into_pocket(th.PRKeys.prior, m.Input(shape, name='prior'))

  return model


def get_loss():
  if 'xmae' not in th.loss_string: return th.loss_string
  from tframe.core.quantity import Quantity

  def kernel(y_true: tf.Tensor, y_predict: tf.Tensor):
    st, sp = [tf.reduce_sum(t) for t in (y_true, y_predict)]
    delta = tf.abs(y_true - y_predict)

    assert 0 < th.alpha < 1

    # (1) penalize FN
    loss_fn = tf.reduce_sum(delta * y_true) / st

    # (2) penalize FP
    loss_fp = tf.reduce_sum(delta * y_predict) / sp

    return th.alpha * loss_fn + (1.0 - th.alpha) * loss_fp

  return Quantity(kernel, name='XMAE')


def finalize(model, squish=True):
  assert isinstance(model, m.Predictor)
  if squish:
    model.add(m.Conv2D(1, 1, use_bias=False, activation='sigmoid'))

  # Build model
  metrics = ['wmae:0.0', 'mae']
  # if th.loss_function is not 'mae': metrics.append('mae')
  # else: metrics.append('wmae:0.0')
  model.build(loss=get_loss(), metric=metrics)
  # model.build(metric=['mse'], batch_metric='mse', eval_metric='mse')
  return model


def conv(filters=None, kernel_size=None, activation=None, strides=1,
         use_batchnorm=None):

  if filters is None: filters = th.filters
  if kernel_size is None: kernel_size = th.kernel_size
  if use_batchnorm is None: use_batchnorm = th.use_batchnorm
  if activation is None:
    if th.activation is not None: activation = th.activation
    else: activation = 'relu' if th.relu_leak == 0 else 'lrelu:{}'.format(
      th.relu_leak)

  return m.Conv2D(filters, kernel_size, strides, use_bias=False,
                  activation=activation, use_batchnorm=use_batchnorm)


def get_unet(mark=True) -> m.UNet2D:
  bridges = th.bridges
  if bridges not in (None, '-', 'a', 'x'):
    bridges = [int(s) for s in bridges.split(',')]
  if th.activation == 'None': th.activation = None

  unet = m.UNet2D(
    th.filters, activation=th.activation, height=th.half_height,
    thickness=th.thickness, use_maxpool=th.use_maxpool,
    use_batchnorm=th.use_batchnorm,
    contraction_kernel_size=th.contraction_kernel_size,
    expansion_kernel_size=th.expansion_kernel_size,
    bottle_neck_after_bridge=th.bottle_neck,
    use_duc=th.use_duc, link_indices=bridges, guest_first=th.guest_first)

  if mark: th.mark += '({})'.format(str(unet))
  return unet







