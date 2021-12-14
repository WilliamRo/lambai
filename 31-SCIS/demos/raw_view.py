import cv2
import os
from roma import console
from lambo import DaVinci



class RawViewer(DaVinci):

  def __init__(self, folder_path, str_fmt):
    # Call parent's constructor
    super().__init__('RawViewer', 6, 6, init_as_image_viewer=True)

    self.str_fmt = str_fmt
    self._init_frames(folder_path)


  def _init_frames(self, path):
    assert os.path.exists(path)

    for i in range(20):
      fn = os.path.join(path, self.str_fmt.format(i + 1))
      assert os.path.exists(fn)
      im = cv2.imread(fn, cv2.IMREAD_ANYDEPTH)
      self.add_image(im)

    console.show_status(f'{len(self.objects)} frames has been read.')



if __name__ == '__main__':
  path = r'../data/yanping-data'
  assert os.path.exists(path)

  str_fmt = r'20210706_2d_rbcs_C001H001S00010000{:02d}.tif'

  cv = RawViewer(path, str_fmt)
  cv.show()

