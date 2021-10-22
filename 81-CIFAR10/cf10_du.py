import numpy as np
import tframe as tfr
from tframe import DataSet
from tframe.data.images.cifar10 import CIFAR10
from tframe.data.augment.img_aug import image_augmentation_processor



def load_data(path):
  datasets = CIFAR10.load(
    path, train_size=None, validate_size=5000, test_size=10000,
    flatten=False, one_hot=True)
  datasets = list(datasets)
  # Get subclasses if required
  if 'dc' in tfr.hub.developer_code:
    assert not 'mg' in tfr.hub.developer_code
    for i in range(len(datasets)):
      datasets[i] = datasets[i].get_classes(3, 5)
    tfr.hub.class_indices = '-'
    tfr.hub.num_classes = 2
  elif 'mg' in tfr.hub.developer_code:
    for i in range(len(datasets)):
      datasets[i].merge_classes(3, 5)
    tfr.hub.class_indices = '-'
    tfr.hub.num_classes = 9

  train_set, val_set, test_set = datasets
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  # Set batch_preprocessor for augmentation if required
  if tfr.hub.augmentation:
    train_set.batch_preprocessor = image_augmentation_processor
  return train_set, val_set, test_set


def evaluate(trainer):
  from cf10_core import th
  from tframe import Classifier
  from tframe.trainers.trainer import Trainer
  model = trainer.model
  assert isinstance(trainer, Trainer) and isinstance(model, Classifier)
  agent = model.agent

  # Get datasets
  ds_dict = {'Train': trainer.training_set, 'Val': trainer.validation_set,
             'Test': trainer.test_set}

  for name, data_set in ds_dict.items():
    assert isinstance(data_set, DataSet)
    cm = model.evaluate_pro(data_set, th.eval_batch_size)
    metric_title = '{} F1'.format(name)
    agent.put_down_criterion(metric_title, cm.macro_F1)
    agent.take_notes('Confusion Matrix on {} Set:'.format(name), False)
    agent.take_notes(cm.matrix_table().content, False)
    agent.take_notes('Evaluation Result on {} Set:'.format(name), False)
    agent.take_notes(cm.make_table(groups=th.class_index_list).content, False)

    # Add cm to note for future analysis
    agent.add_to_note_misc('{} CM'.format(name), cm)

    # Additional information for test set
    if name == 'Test' and th.class_index_list:
      class_names = data_set.properties['CLASSES']
      indices = th.class_index_list
      key = '/'.join([class_names[i].upper()[0] for i in indices])
      # Class F1
      F1 = np.average(cm.F1s[np.array(indices)])
      agent.put_down_criterion('Test {} F1'.format(key), F1)
      agent.take_notes('{} F1 on Test Set: {}'.format(
        key, th.decimal_str(F1)), False)
      # Exchange
      total = sum([cm.confusion_matrix[i, j] for i in indices for j in indices])
      num_exchange = (cm.confusion_matrix[indices[0], indices[1]]
                      + cm.confusion_matrix[indices[1], indices[0]])
      ratio = num_exchange / total
      agent.put_down_criterion('Test {} Exchange Ratio'.format(key), ratio)
      agent.take_notes(
        '{} Exchange Ratio: {}'.format(key, th.decimal_str(ratio)), False)


if __name__ == '__main__':
  from cf10_core import th
  from tframe.data.images.image_viewer import ImageViewer
  train_set, val_set, test_set = load_data(th.data_dir)
  # viewer = ImageViewer(test_set.get_classes(3, 5))
  # test_set.merge_classes(3, 5)
  viewer = ImageViewer(test_set)
  viewer.show()


