from tframe.data.base_classes import DataAgent
import os
from spr.spr_set import SPRSet



class SPRAgent(DataAgent):

  @classmethod
  def load(cls, data_dir):
    data_set = cls.load_as_tframe_data(data_dir)
    from spr_core import th
    if th.folds_k is None: data_sets = data_set.split(
      -1, th.val_size, th.test_size, names=('Train Set', 'Val Set', 'Test Set'))
    else: data_sets = data_set.split_k_fold(th.folds_k, th.folds_i)


    return data_sets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs):
    data_path = os.path.join(data_dir, 'simulate_fringe.tfdir')
    if os.path.exists(data_path):
      return SPRSet.load(data_path)
    else:
      data_dict = cls.load_as_numpy_arrays(data_dir)
      ps = SPRSet(data_dict=data_dict, name='simulate_fringe')
      ps.save(data_path)
      return ps


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    from ditto import Painter, Config
    from spr_core import th
    import numpy as np

    data_dict = {}
    data_dict[SPRSet.FEATURES] = []
    data_dict[SPRSet.TARGETS] = []
    p = Painter(**Config)
    from roma import console
    console.show_status('Generating Data...')
    for i in range(th.data_size):
      p.paint_samples()
      data_dict[SPRSet.FEATURES].append(np.expand_dims(p.extracted_fringe, axis=-1))
      data_dict[SPRSet.TARGETS].append(np.expand_dims(p.img, axis=-1))
      console.print_progress(i, th.data_size)
    console.show_status('Data finished.')
    return data_dict


if __name__ == '__main__':
  from spr_core import th
  print(th.data_dir)
  #PetAgent.load_as_numpy_arrays(data_dir=th.data_dir)
  ps = SPRAgent.load_as_tframe_data(th.data_dir)
  ps.visualize_pet()
