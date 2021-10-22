import tensorflow as tf
from tframe import Classifier

from tframe.layers import Input, Activation
from tframe.layers.common import Dropout
from tframe.models import Recurrent

# from tframe.configs.config_base import Config
from tframe.trainers.trainer import TrainerHub as Config

from tframe.layers.common import Onehot
from tframe.layers.advanced import Dense
from tframe.layers.embedding import Embedding


def typical(th, cells):
  assert isinstance(th, Config) and th.hidden_dim > 0

  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)

  # Add layers
  model.add(Input(sample_shape=th.input_shape, dtype=tf.int32))
  emb_init = tf.initializers.random_uniform(-0.25, 0.25)
  model.add(Embedding(th.num_words, th.hidden_dim, initializer=emb_init))

  if th.input_dropout > 0: model.add(Dropout(1 - th.input_dropout))
  # Add hidden layers
  if not isinstance(cells, (list, tuple)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  output_and_build(model, th)
  return model


def output_and_build(model, th):
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)
  # Add dropout if necessary
  if th.output_dropout > 0: model.add(Dropout(1 - th.output_dropout))
  # Add output layer
  model.add(Dense(
    num_neurons=th.output_dim,
    prune_frac=0.5,
  ))
  model.add(Activation('sigmoid', set_logits=True))

  model.build(last_only=True, loss='sigmoid_cross_entropy',
              metric=['loss', 'accuracy'],
              batch_metric='accuracy', eval_metric='accuracy')


