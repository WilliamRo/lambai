import numpy as np
from tframe import console
from tframe.utils import misc
from tframe.data.shadow import DataShadow
from tframe.data.images.ire_img import IrregularImageSet
from lambo import DaVinci


class Visualizer(DaVinci):
  def __init__(self):
    super(Visualizer, self).__init__('Fringe Visualizer')

  def show_img(self, x):
    self.imshow(np.sum(x[0], axis=-1))

  def show_fringe(self, x):
    self.imshow(np.sum(x[1], axis=-1))


class SPRSet(IrregularImageSet):
  @staticmethod
  def wmae(truth: np.ndarray, pred: np.ndarray):
    return np.sum(np.abs(truth - pred) * truth) / np.sum(truth)

  def visualize(self):
    da = Visualizer()
    da.objects = np.array([self.data_dict[self.FEATURES],
                                 self.data_dict[self.TARGETS]])
    da.objects = np.transpose(da.objects,axes=(1, 0, 2, 3, 4))
    da.add_plotter(da.show_img)
    da.add_plotter(da.show_fringe)
    da.show()

  def snapshot(self, model, index=0, over_trial=False, step='final-'):
    indices = [index]
    for i in indices: self._snapshot(
      model, i, save_input=False, save_ground_truth=True, step=step)

  def _snapshot(self, model, index=0, folder_path=None, save_input=False,
                save_ground_truth=True, step=''):
    from tframe import Predictor
    from pr_core import th
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)
    y = model.predict(self[index], batch_size=1, verbose=False)
    y = np.reshape(y, y.shape[1:3])
    gt = self.targets[index].reshape(y.shape)
    if th.feature_type == 1:
      x = self.features[index].reshape(y.shape)
    else:
      x = self.features[index][:, :, 0].reshape(y.shape)

    # Save input (if required), prediction and ground truth
    if folder_path is None: folder_path = model.agent.ckpt_dir
    suffix = '-{}-{}.png'.format(self.name, index)

    metric_str = '({:.4f})'.format(self.wmae(gt, y))
    for name, flag, img in zip(
        ('input', 'ground-truth', step + 'predicted' + metric_str),
        (save_input, save_ground_truth, True), (x, gt, y)):
      if not flag: continue
      path = os.path.join(folder_path, name + suffix)
      if not os.path.exists(path): plt.imsave(path, img)