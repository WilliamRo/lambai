import cv2
import os
from roma import console
from lambo import DaVinci



class CytoViewer(DaVinci):

  def __init__(self, video_path):
    # Call parent's constructor
    super().__init__('CytoViewer', 6, 8, init_as_image_viewer=True)

    self._init_frames(video_path)



  def _init_frames(self, path):
    assert os.path.exists(path)

    cap = cv2.VideoCapture(path)
    assert cap.isOpened()

    console.show_status('Reading frames ...')
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret: break
      # See https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
      self.add_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    console.show_status(f'{len(self.objects)} frames has been read.')
    cap.release()



if __name__ == '__main__':
  path = r'../data/cyt01.mp4'
  assert os.path.exists(path)

  cv = CytoViewer(path)
  cv.show()

