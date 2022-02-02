import numpy as np

from pr.pr_agent import PRAgent
from pr.pr_set import PhaseSet

from typing import List


def load_data() -> [PhaseSet, PhaseSet, PhaseSet]:
  from pr_core import th

  # Load datasets normally
  datasets = PRAgent.load(
    th.data_dir, *th.train_val_test_indices(), radius=th.radius,
    win_size=th.win_size, truncate_at=th.truncate_at, win_num=th.win_num,
    fn_pattern=th.fn_pattern, re_pattern=th.re_pattern,
    random_rotate=th.random_rotate, feature_type=th.feature_type)

  # Call dev0 manipulator if required
  if 'dev0' in th.pr_dev_code: datasets = _dev0_loader(datasets)

  # Apply configs and report detail
  for i, config in enumerate((th.train_config, th.val_config, th.test_config)):
    # Apply config if required
    if config not in (None, '', 'x', '-'):
      datasets[i] = datasets[i].get_subset_by_config_str(config)
    # Report detail
    datasets[i].report_data_details()

  # Return datasets
  train_set, val_set, test_set = datasets

  # Do something special
  if 'jan29' in th.developer_code:
    test_set = test_set[list(range(1, 15)) + [39]]

  return train_set, val_set, test_set


def _dev0_loader(datasets: List[PhaseSet]) -> List[PhaseSet]:
  from pr_core import th

  train_set = datasets[0]
  index = int(th.pr_dev_code[-1])
  return [train_set[index]] * 3


if __name__ == '__main__':
  from pr_core import th

  th.train_indices = '1'
  th.val_indices = '1'
  th.test_indices = '1'

  # th.fn_pattern = '0[45]-'
  th.fn_pattern = '01-'
  # th.fn_pattern = '02-'
  # th.fn_pattern = '*62-'

  th.feature_type = 1

  # th.train_config = '2a3'
  # th.train_config = '39a40'
  # th.val_config = '10:15'
  # th.test_config = '15:'

  # th.edge_cut = 8
  # th.developer_code = 'jan29'

  train_set, val_set, test_set = load_data()
  assert isinstance(train_set, PhaseSet)
  assert isinstance(val_set, PhaseSet)
  assert isinstance(test_set, PhaseSet)
  # test_set.view_aberration()
  # train_set.view()
  # train_set.test_window(512, 10)
  # val_set.view()
  test_set.view()

  # win_num = 10
  # win_size = 512
  # test_set.test_window(win_size, win_num)

