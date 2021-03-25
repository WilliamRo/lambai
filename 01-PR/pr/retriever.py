"""This module provides a class for 2-D phase retrieval.

Class Design Criteria:
(1) Private attributes are not used if not necessary.
"""
import cv2
import numpy as np

from lambo.data_obj import DigitalImage


class Retriever(object):

  def __init__(self, img, bg=None):
    # Check input type
    assert isinstance(img, DigitalImage)
    if bg is not None: assert isinstance(bg, DigitalImage)
    self.img = img
    self.bg = bg

  # region: Properties

  # endregion: Properties

  # region: Public Methods

  def get_mask(self, radius):
    assert isinstance(radius, int)

    return None

  # endregion: Public Methods

  # region: Methods for Display

  def show_images(self, img=False, masked_Sc=False):
    imgs = []

  # endregion: Methods for Display


if __name__ == '__main__':
  import os
  folder_path = r'../data/3t3'
  img, bg = [DigitalImage.imread(os.path.join(folder_path, p, '1.tif'))
             for p in ('sample', 'bg')]
  assert isinstance(img, DigitalImage)
  assert isinstance(bg, DigitalImage)
  # img.imshow(True)
  bg.imshow(True)
  r = Retriever(img, bg)
