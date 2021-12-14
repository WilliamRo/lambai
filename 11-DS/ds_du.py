import numpy as np
import tframe as tfr
from tframe import DataSet

from pet.pet_agent import PetAgent
from pet.pet_set import PetSet



def load_data(path, val_size=2000, test_size=2000):
  train_set, val_set, test_set = PetAgent.load(path, val_size, test_size)

  return train_set, val_set, test_set


if __name__ == '__main__':
  from pet_core import th
  train_set, val_set, test_set = load_data(th.data_dir)
  assert isinstance(val_set, PetSet)
  # train_set.visualize()
  # val_set.visualize()
  test_set.visualize()


