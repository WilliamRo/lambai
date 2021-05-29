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
    fn_pattern=th.fn_pattern, random_rotate=th.random_rotate,
    feature_type=th.feature_type)

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
  th.test_indices = '2'
  th.fn_pattern = '0[45]-'
  # th.fn_pattern = '*62-'

  th.feature_type = 9

  # th.train_indices = '4'
  # th.val_indices = '4'
  # th.test_indices = '4'
  # th.train_config = ':10'
  # th.val_config = '10:15'
  # th.test_config = '15:'

  # th.pr_dev_code = 'dev.0'
  # th.prior_size = 12
  # th.use_prior = True

  train_set, val_set, test_set = load_data()
  assert isinstance(train_set, PhaseSet)
  assert isinstance(val_set, PhaseSet)
  assert isinstance(test_set, PhaseSet)
  # test_set.view_aberration()
  train_set.view()

  # win_num = 10
  # win_size = 512
  # test_set.test_window(win_size, win_num)

