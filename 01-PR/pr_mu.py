import tensorflow as tf

from tframe import mu as m
from pr_core import th


def get_container():
  model = m.Predictor(mark=th.mark)
  model.add(m.Input(sample_shape=th.input_shape))
  return model


def finalize(model, squish=True):
  assert isinstance(model, m.Predictor)
  if squish: model.add(m.Conv2D(1, 1, use_bias=True))
  # Build model
  model.build(metric='mse', eval_metric='mse')
  # model.build(metric=['mse'], batch_metric='mse', eval_metric='mse')
  return model


