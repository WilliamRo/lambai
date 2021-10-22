import os
import numpy as np
from tframe.utils.note import Note


notes_and_filters = {
  r'E:\lambai\02-WBC\00_seed\0225_s2_seed_s02.sum': {
    'archi_string': '12-16-24-32=32-16',
    'learning_rate': 0.0003,
    'use_wise_man': True,
    'activation': 'relu',
  },
  r'E:\lambai\02-WBC\00_seed\0226_s2_seed_s05.sum': {
    'archi_string': '12-16-24-32=32-16',
    'learning_rate': 0.0001,
    'use_wise_man': True,
    'activation': 'relu',
    'batch_size': 48,
    'patience': 10,
  },
  r'E:\lambai\02-WBC\02_lenet\0226_s2_lenet_BT.sum': {
    'patience': 7,
    'only_BT': True,
  },
}


def report(notes_, key):
  assert key in ('Train', 'Val', 'Test')
  cms = [n.misc['{} CM'.format(key)] for n in notes_]
  print('precision/recall/F1 on {} Set (support: {})'.format(
    key, len(cms)))
  def _report_type(name, index):
    s = name + ': '
    try:
      for c in ('precisions', 'recalls', 'F1s'):
        vals = [getattr(cm, c)[index] for cm in cms]
        s += '{:.3f}+-{:.3f}'.format(np.mean(vals), np.std(vals))
        s += '/' if c != 'F1s' else ''
    except: return
    print(s)
  for i, name in enumerate(('B-Cell', 'T-Cell', 'Granu', 'Mono')):
    _report_type(name, i)


for path, filter_dict in notes_and_filters.items():
  notes = Note.load(path)
  # Filter
  notes = [n for n in notes if all(
    [n.configs[k] == v for k, v in filter_dict.items()])]

  # Calculate and print
  print(os.path.basename(path))
  for key in ('Train', 'Val', 'Test'): report(notes, key)

  print()