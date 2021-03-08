from tframe import console
from tframe import pedia
from tframe.data.images.ire_img import IrregularImageSet
from tframe.utils.misc import convert_to_one_hot
from tframe.utils.display.table import Table
from tframe.utils.np_tools import pad_or_crop

import numpy as np


class BloodCellSet(IrregularImageSet):

  DONOR_KEY = 'donors'
  DONOR_NAME_KEY = 'donor_list'
  DENSE_LABEL_KEY = 'dense_labels'
  EXTENSIONS = ['tfd', 'tfdir']

  # region: Properties

  @property
  def with_donors(self): return self.DONOR_KEY in self.data_dict


  @property
  def donor_names(self): return self.properties[self.DONOR_NAME_KEY]


  @property
  def donors(self): return self.data_dict[self.DONOR_KEY]
  
  
  @property
  def donor_groups(self):
    if self.DONOR_KEY not in self.data_dict: return None
    groups = []
    for i, _ in enumerate(self.donor_names):
      groups.append(list(np.argwhere(self.donors == i).ravel()))
    return groups

  # endregion: Properties

  # region: Public Methods

  def get_types(self, class_indices):
    """Get specific types of cell images"""
    assert isinstance(class_indices, (tuple, list))
    indices = []
    for i in class_indices: indices += self.groups[i]
    data_set = self[indices]
    # Set corresponding properties
    data_set.properties[self.NUM_CLASSES] = 2
    # data_set is in one-hot format
    data_set.targets = data_set.targets[:, np.array(class_indices)]
    # Set group names
    group_names = data_set.properties[pedia.classes]
    data_set.properties[pedia.classes] = [group_names[i] for i in class_indices]
    data_set.refresh_groups()
    return data_set


  def separate(self, num_or_indices, random=True, over_classes=True,
               name1='separated_set', name2='remaining_set'):
    """Separate dataset by donor indices or images count.

    SYNTAX:
        # Separate donor 2 from this dataset
        separated, remaining = separate((2,))
        # Separate 100 cell images for each cell type randomly
        separated, remaining = separate(100, random=True, over_classes=True)

    :param num_or_indices: number of cell images to separate (positive integer)
                           or donor indices (tuple or list)
    :param random: whether to separate randomly
                  (take effect if 'num_or_indices' is int)
    :param over_classes: whether to separate over classes
                  (take effect if 'num_or_indices' is int)
    :param name1: name of separated set
    :param name2: name of remaining set
    :return: the specified data_set and the remaining data_set
    """
    # Define sampling rule
    def _sample(total, n):
      if random: return np.random.choice(total, n, replace=False)
      # Non-random sampling rule
      return np.arange(total)[slice(0, None, total // n)][:n]

    # Generate indices accordingly
    image_indices = []
    if isinstance(num_or_indices, (tuple, list)):
      # Separate by donor indices
      donor_indices = num_or_indices
      donor_groups = self.donor_groups
      for index in donor_indices:
        # Make sure self contains cell images from donor[index]
        if len(donor_groups[index]) == 0: raise AssertionError(
          '!! {} does not contain cell images from donor {}'.format(
            self.name, index + 1))
        image_indices += donor_groups[index]
    elif over_classes:
      # Separate by image number of each class
      num = num_or_indices
      for i, g in enumerate(self.groups):
        g = np.array(g)
        # Make sure each class has at least $num$ samples
        if len(g) < num: raise AssertionError(
          '!! Not enough ({}) {} images.'.format(num, self[pedia.classes][i]))
        image_indices += list(g[_sample(len(g), num)])
    else:
      # Separate by image number
      num = num_or_indices
      if num > self.size: raise AssertionError(
        '!! Not enough images ({}) in {} to separate'.format(num, self.name))
      image_indices = _sample(self.size, num)

    # Generate remaining indices
    remain_indices = [i for i in range(self.size) if i not in image_indices]
    if len(remain_indices) == 0: raise AssertionError(
      '!! No cell images remaining')
    # Split and return
    return self.split_by_indices(
      (image_indices, remain_indices), (name1, name2))


  def split_by_indices(self, indices_list, names):
    assert len(indices_list) == len(names)
    data_sets = ()
    for indices, name in zip(indices_list, names):
      data_set = self[indices]
      data_set.name = name
      data_set.refresh_groups()
      data_sets += (data_set,)
    return data_sets


  def preprocess(self, H=350, W=320, pad_mode='constant'):
    from tframe.utils.display.progress_bar import ProgressBar
    # Preprocess one by one
    bar = ProgressBar(total=self.size)
    console.show_status('Preprocessing ...')
    for i, x in enumerate(self.features):
      # Shift each image so the min(image) is 0
      x -= np.min(x)
      # :: Pad or crop image
      # (1) Height
      x = pad_or_crop(x, axis=0, size=H, pad_mode=pad_mode)
      # (2) Width
      x = pad_or_crop(x, axis=1, size=W, pad_mode=pad_mode)
      self.features[i] = x
      # Show progress bar
      bar.show(i + 1)
    self.features = np.stack(self.features)
    # Convert label to one-hot tensors
    console.show_status('Converting dense labels to one-hot tensors ...')
    self.dense_labels = self.targets
    self.targets = convert_to_one_hot(self.targets, self.num_classes)
    console.show_status('Preprocess completed.')

  # endregion: Public Methods

  # region: Display

  def view(self, shuffle=False):
    from tframe.data.images.image_viewer import ImageViewer
    from matplotlib import cm
    # Scale images to [0, 1]
    data_set = self[:]
    data_set.features = [x - np.min(x) for x in data_set.features]
    data_set.features = [x / np.max(x) for x in data_set.features]
    # Shuffle if necessary
    if shuffle:
      indices = list(range(self.size))
      np.random.shuffle(indices)
      data_set.features = [data_set.features[i] for i in indices]
      data_set.targets = data_set.targets[indices]
    # Show data using ImageViewer
    viewer = ImageViewer(data_set, color_map=cm.gist_earth)
    viewer.show()


  def report_data_details(self):
    # Report data details in a table
    cell_names = self.properties[pedia.classes]
    width = max([min(len(c), 8) for c in cell_names])
    t = Table(9, *([width] * self.num_classes), tab=3, margin=1, buffered=False)
    t.specify_format(align='l' + 'r'*self.num_classes)
    t.print(':: Details of {}'.format(self.name))
    t.print_header('Donor' if self.with_donors else '', *cell_names)

    # Print details for each donor if donor information exists
    donor_count = 0
    if self.with_donors:
      donor_groups = self.donor_groups
      for donor_id, donor in enumerate(self.donor_names):
        if len(donor_groups[donor_id]) == 0: continue
        cells = ['{}/{} {}'.format(donor_id + 1, len(self.donor_names), donor)]
        for i, cell_name in enumerate(self.properties[pedia.classes]):
          # Get image # for each cell type
          cells.append(
            len([k for k in range(self.size)
                 if self.dense_labels[k] == i and self.donors[k] == donor_id]))
        t.print_row(*cells)
        donor_count += 1
    # Split
    t.hline()

    # Show overall information if necessary
    if any([self.with_donors and donor_count > 1, not self.with_donors]):
      t.print_row('Overall', *[len(g) for g in self.groups])
      t.hline()

    # Get size detail
    if isinstance(self.features, list):
      Hs, Ws = [[img.shape[i] for img in self.features] for i in (0, 1)]
      H_max, H_min, W_max, W_min = (
        np.max(Hs), np.min(Hs), np.max(Ws), np.min(Ws))
      size_detail = 'Max HxW: {}x{}; Min HxW: {}x{}'.format(
        H_max, W_max, H_min, W_min)
    else:
      assert isinstance(self.features, np.ndarray)
      shape = self.features.shape
      assert len(shape) == 3
      size_detail = 'Image shape: {}x{}'.format(shape[1], shape[2])

    # Show summaries
    t.print('{}Totally {} images; '.format(
      ' ' * t._margin, self.size) + size_detail)
    # End of the table
    t.hline()

  # endregion: Display

  # region: Methods Overriding

  @classmethod
  def load(cls, filename):
    bcs = super().load(filename)
    assert isinstance(bcs, BloodCellSet)
    bcs.report_data_details()
    return bcs

  # endregion: Methods Overriding

  # region: Evaluation

  @staticmethod
  def evaluate(trainer):
    from tframe import hub as th
    from tframe import Classifier
    from tframe.trainers.trainer import Trainer
    model = trainer.model
    assert isinstance(trainer, Trainer) and isinstance(model, Classifier)
    agent = model.agent

    # Get test set
    ds_dict = {'Train': trainer.training_set, 'Val': trainer.validation_set,
               'Test': trainer.test_set}

    for name, data_set in ds_dict.items():
      assert isinstance(data_set, BloodCellSet)
      cm = model.evaluate_pro(data_set, th.eval_batch_size)
      metric_title = '{} F1'.format(name)
      agent.put_down_criterion(metric_title, cm.macro_F1)
      agent.take_notes('Confusion Matrix on {} Set:'.format(name), False)
      agent.take_notes(cm.matrix_table().content, False)
      agent.take_notes('Evaluation Result on {} Set:'.format(name), False)
      agent.take_notes(cm.make_table().content, False)

      # Add cm to note for future analysis
      agent.add_to_note_misc('{} CM'.format(name), cm)

      # Additional information for test set
      if name == 'Test':
        # BT_F1
        BT_F1 = np.average(cm.F1s[:2])
        agent.put_down_criterion('Test B/T F1', BT_F1)
        agent.take_notes('B/T Cell F1 on Test Set: {}'.format(
          th.decimal_str(BT_F1)), False)
        # BT Exchange
        total = sum([cm.confusion_matrix[i, j] for i in (0, 1) for j in (0, 1)])
        num_exchange = cm.confusion_matrix[0, 1] + cm.confusion_matrix[1, 0]
        ratio = num_exchange / total
        agent.put_down_criterion('Test B/T Exchange Ratio', ratio)
        agent.take_notes(
          'B/T Exchange Ratio: {}'.format(th.decimal_str(ratio)), False)

  # endregion: Evaluation
