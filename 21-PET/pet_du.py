import numpy as np
import tframe as tfr
from tframe import DataSet

from pet.pet_agent import PetAgent
from pet.pet_set import PetSet



def load_data(path):
  return PetAgent.load(path)



if __name__ == '__main__':
  from pet_core import th

  th.folds_k = 5
  th.folds_i = 2

  if th.folds_k is None:
    train_set, val_set, test_set = load_data(th.data_dir)
    assert isinstance(val_set, PetSet)
    # train_set.visualize()
    # val_set.visualize()
    test_set.visualize()
  else:
    train_set, val_set = load_data(th.data_dir)
    assert isinstance(val_set, PetSet)
    val_set.visualize()


