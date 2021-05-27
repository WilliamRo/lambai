import tensorflow as tf

from tframe import context
from tframe import Predictor
from tframe import losses
from tframe import mu as m
from pr_core import th

import pr.architectures.konjac as konjac


def get_container():
  model = m.Predictor(mark=th.mark)
  model.add(m.Input(sample_shape=th.input_shape))

  if th.use_prior:
    assert isinstance(th.prior_size, int) and th.prior_size > 0
    shape = [th.prior_size, th.prior_size, 2]
    if th.prior_key == 'dettol':
      coef = (2 if th.prior_format in ('dc', 'doulbe_channel', 'default')
              else 1)
      shape[-1] = int(len(th.kon_rs) * 180 / th.kon_omega) * coef
    context.put_into_pocket(th.PRKeys.prior, m.Input(shape, name='prior'))

  return model


def finalize(model, squish=True):
  assert isinstance(model, m.Predictor)
  if squish:
    model.add(m.Conv2D(1, 1, use_bias=False, activation='sigmoid'))

  # Build model
  metrics = ['wmae:0.0', 'mae']
  # if th.loss_function is not 'mae': metrics.append('mae')
  # else: metrics.append('wmae:0.0')
  model.build(loss=th.loss_string, metric=metrics)
  # model.build(metric=['mse'], batch_metric='mse', eval_metric='mse')
  return model




