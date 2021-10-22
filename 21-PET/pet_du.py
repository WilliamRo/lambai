import numpy as np
import tframe as tfr
from tframe import DataSet

from pet.pet_agent import PetAgent



def load_data(path, val_size, test_size):
  train_set, val_set, test_set = PetAgent.load(path, val_size, test_size)

  return train_set, val_set, test_set


if __name__ == '__main__':
  # from pet_core
  train_set, val_set, test_set = PetAgent.load(
    r'E:\lambai\21-PET\data', None, None)


