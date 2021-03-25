import numpy as np

from pr.pr_agent import PRAgent
from pr.pr_set import PhaseSet


def load_data():
  from pr_core import th

  if 'dev.0' in th.pr_dev_code: return _dev0_loader()

  datasets = PRAgent.load(
    th.data_dir, *th.train_val_test_indices(), radius=th.radius)
  for ds in datasets: ds.report_data_details()
  return datasets


def _dev0_loader():
  from pr_core import th

  datasets = PRAgent.load(th.data_dir, 1, 2, 3, radius=th.radius)
  for ds in datasets: ds.add_channel()
  datasets = [ds[0] for ds in datasets]
  N = 100
  train_set = datasets[0]
  train_set.features = np.concatenate([train_set.features] * N)
  train_set.targets = np.concatenate([train_set.targets] * N)
  train_set.data_dict['interferograms'] = train_set['interferograms'] * N
  return datasets


if __name__ == '__main__':
  from pr_core import th

  th.train_indices = '1,2'
  th.val_indices = '3'
  th.test_indices = '4'

  th.pr_dev_code = 'dev.0'

  train_set, val_set, test_set = load_data()
  train_set.view()
