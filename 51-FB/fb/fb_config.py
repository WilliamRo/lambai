from tframe.configs.config_base import Flag
from tframe.trainers import SmartTrainerHub


class FBConfig(SmartTrainerHub):

  fb_data_size = Flag.integer(100, 'Dataset.size', is_key=None)
  fb_img_size = Flag.integer(100, 'FB image size', is_key=None)
  fb_min_size = Flag.integer(5, 'Minimum size', is_key=None)
  fb_max_size = Flag.integer(20, 'Maximum size', is_key=None)
  fb_min_boxes = Flag.integer(5, 'Minimum box number', is_key=None)
  fb_max_boxes = Flag.integer(15, 'Maximum box number', is_key=None)
  fb_shape = Flag.string('square', 'Object shape', is_key=None)
  fb_max_ovlp = Flag.float(0.2, 'Max overlap IoU', is_key=None)

  yolo_S = Flag.integer(7, 'S in YOLO', is_key=None)
  yolo_B = Flag.integer(2, 'B in YOLO', is_key=None)
  yolo_D = Flag.integer(5, 'D in YOLO', is_key=None)

  yolo_coor = Flag.float(5.0, 'lambda_coordinate', is_key=None)
  yolo_noob = Flag.float(0.5, 'lambda_no_object', is_key=None)

  data_converter = Flag.whatever(None, 'Data converter')
  pred_converter = Flag.whatever(None, 'Prediction converter')

  min_confidence = Flag.float(0.1, 'Minimum confidence', is_key=None)

  visualize_after_training = Flag.boolean(True, '...')

  def set_data(self, token='alpha'):
    from roma import console

    if token in ('alpha', 'a'):
      self.fb_img_size = 64
      self.fb_min_size = 10
      self.fb_max_size = 20
      self.fb_min_boxes = 5
      self.fb_max_boxes = 10
      self.developer_code += '-dup'
      console.show_status('Data set to `alpha`')
    elif token in ('beta', 'b'):
      self.fb_data_size = 20
      self.val_size = 10

      self.fb_img_size = 32
      self.fb_min_size = 5
      self.fb_max_size = 10
      self.fb_min_boxes = 1
      self.fb_max_boxes = 3

      self.developer_code += 'beta'
      console.show_status('Data set to `beta`')
    elif token in ('gamma', 'g'):
      self.fb_data_size = 1000
      self.val_size = 100

      self.fb_img_size = 32
      self.fb_min_size = 5
      self.fb_max_size = 10
      self.fb_min_boxes = 1
      self.fb_max_boxes = 3

      self.developer_code += 'gamma'
      console.show_status('Data set to `gamma`')

# New hub class inherited from SmartTrainerHub must be registered
FBConfig.register()
