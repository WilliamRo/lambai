import os
import numpy as np
from tframe.utils.note import Note
from tframe.utils.maths.confusion_matrix import ConfusionMatrix

import sklearn
from sklearn.metrics import ConfusionMatrixDisplay


notes_and_filters = {
  r'E:\lambai\02-WBC\02_lenet\0226_s2_lenet_BT.sum': {
    'patience': 7,
    'only_BT': False,
  },
}

for path, filter_dict in notes_and_filters.items():
  notes = Note.load(path)
  # Filter
  notes = [n for n in notes if all(
    [n.configs[k] == v for k, v in filter_dict.items()])]

  note = sorted(notes, key=lambda n: n.misc['Val CM'].F1s[0])[0]
  cm = note.misc['Val CM']
  assert isinstance(cm, ConfusionMatrix)

  # Display
  cm.sklearn_plot()


