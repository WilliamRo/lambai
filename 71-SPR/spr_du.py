from spr.spr_agent import SPRAgent



def load_data(path):
  train_set, val_set, test_set = SPRAgent.load(path)

  return train_set, val_set, test_set


if __name__ == '__main__':
  from spr_core import th
  train_set, val_set, test_set = load_data(th.data_dir)
  train_set.visualize()