from tframe import DataSet

from scis.scis_agent import SCISAgent
from scis.scis_set import SCISet



def load_data(path):
  return SCISAgent.load(path)



if __name__ == '__main__':
  from scis_core import th

  th.cell_type = 'c'

  data_set = load_data(th.data_dir)
  data_set.visualize()
