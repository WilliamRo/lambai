from tframe import checker
from tframe.data.sequences.nlp.imdb import IMDB
from tframe.data.sequences.seq_set import SequenceSet


def load_data(path, train_size=15000, val_size=10000, test_size=25000,
              num_words=10000, max_len=512):
  data_sets = IMDB.load(
    path, train_size, val_size, test_size, num_words, max_len)
  checker.check_type(data_sets, SequenceSet)
  return data_sets


if __name__ == '__main__':
  from imdb_core import th
  data_sets = load_data(th.data_dir)
  print()
