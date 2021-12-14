import numpy as np
from tframe import console
from tframe.utils import misc
from tframe.data.shadow import DataShadow
from tframe.data.images.ire_img import IrregularImageSet


class PetSet(IrregularImageSet):

  SHADOW_KEY = 'SHADOW_KEY'
  SIZES_KEY = 'SIZES_KEY'
  MAX_BATCH_SIZE = 500
  META = 'meta'

  # region: Properties

  @IrregularImageSet.features.getter
  def features(self):
    # Case I, resized images has already been saved
    if self.FEATURES in self.data_dict: return self.data_dict[self.FEATURES]

    # Case II, resized images have to be generated online
    if self.size > self.MAX_BATCH_SIZE:
      raise AssertionError(f'Maximum legal size is {self.MAX_BATCH_SIZE}, '
                           f'got {self.size} instead.')

    return self.image_unifier([s.data for s in self.shadows])

  @property
  def shadows(self):
    return self.data_dict[self.SHADOW_KEY]

  @property
  def sizes(self):
    if self.SIZES_KEY not in self.data_dict: return None
    return self.data_dict[self.SIZES_KEY]

  @property
  def meta_keys(self):
    return [k for k in self.data_dict.keys() if k not in (
      self.FEATURES, self.TARGETS, 'id', self.SIZES_KEY, self.SHADOW_KEY)]

  # endregion: Properties

  def __getitem__(self, item):
    if item == self.FEATURES: return self.features
    return super().__getitem__(item)

  # region: Public Methods

  def set_targets(self):
    from pet_core import th

    # Prepare meta data if required
    if th.use_meta_data:
      self.data_dict[self.META] = np.stack(
        [self.data_dict[k] for k in self.meta_keys], axis=1)

    # Set targets accordingly
    if not th.use_classifier:
      self.targets = np.expand_dims(self.targets, 1)
      return

    y = (self.targets - 1) // (100 // th.num_classes)
    self.targets = misc.convert_to_one_hot(y, th.num_classes)

  def report(self):
    console.show_info(f'{self.name} ({self.size} images)')

  def visualize(self):
    from pet.pet_viewer import PetViewer
    pv = PetViewer(self)
    pv.compare_hist()
    pv.show_size_hist()
    pv.show()

  @staticmethod
  def unify(im: np.ndarray):
    import cv2
    from pet_core import th

    top, bottom, left, right = [0] * 4
    H, W = im.shape[:2]
    L = th.pet_input_size
    if H > W:
      H, W = L, int(W / H * L)
      left = (L - W) // 2
      right = L - W - left
    else:
      W, H = L, int(H / W * L)
      top = (L - H) // 2
      bottom = L - H - top
    im = cv2.resize(im, dsize=(W, H))

    return cv2.copyMakeBorder(im, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=0)

  def image_unifier(self, im_list: list):
    from pet_core import th
    if th.pet_input_size is None: return im_list
    return np.stack([self.unify(im) for im in im_list], axis=0)

  # endregion: Public Methods

  # region: Strategy Tests

  def naive(self, c=None):
    if c is None: c = np.mean(self.targets)
    console.show_status(f'Testing c = {c:.1f} ...')
    rmse = np.sqrt(np.mean(np.square(c - self.targets)))
    console.show_info(f'RMSE = {rmse:.2f}')

  # endregion: Strategy Tests


if __name__ == '__main__':
  from pet_core import th
  from pet_agent import PetAgent

  # th.pet_input_size = 256
  ps = PetAgent.load_as_tframe_data(th.data_dir, traverse=True)
  ps.naive()
  # ps.visualize()

