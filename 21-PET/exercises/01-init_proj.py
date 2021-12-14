"""
In this exercise you are asked to create your own solution folder.
This can be done simply by copying other XX_core.py, XX_du.py, and
XX_mu.py files from other solution directories and making corresponding
modifications.
Data should also be downloaded and unzipped into the correct location.

Run this script directly to see whether you have done this right.
Note that no modification should be made to this script.
"""
import sys
# sys.path.insert(0, '../')  # TODO: uncomment to switch to main branch

import os
import numpy as np
from PIL import Image

from lambo import DaVinci
from pet_core import th


da = DaVinci('Exercise 1', init_as_image_viewer=True)
da.add_image(
  np.array(Image.open(os.path.join(
    th.data_dir, r'petfinder-pawpularity-score\train',
    '62681561c493f76be77d0abb0b6fea6b.jpg'))),
  'Well done!')
da.show()
