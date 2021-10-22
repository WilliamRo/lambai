import numpy as np
from tframe import console
from tframe.data.shadow import DataShadow
from tframe.data.images.ire_img import IrregularImageSet


class PetSet(IrregularImageSet):

  @property
  def meta_keys(self):
    return [k for k in self.data_dict.keys() if k not in (
      self.FEATURES, self.TARGETS, 'id')]

  def visualize(self):
    from pet_viewer import PetViewer
    pv = PetViewer(self)
    pv.compare_hist()
    pv.show()


if __name__ == '__main__':
  from pet_core import th
  from pet_agent import PetAgent

  ps = PetAgent.load_as_tframe_data(th.data_dir)
  ps.visualize()

