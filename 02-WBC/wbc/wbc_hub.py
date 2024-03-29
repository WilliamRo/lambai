from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class WBCHub(SmartTrainerHub):

  image_side_length = Flag.integer(
    300, 'Side length of square image', is_key=None)
  image_height = Flag.integer(350, 'Blood cell image height', is_key=None)
  image_width = Flag.integer(320, 'Blood cell image width', is_key=None)
  save_HW_data = Flag.boolean(True, 'Whether to save preprocessed data')

  with_donor = Flag.boolean(
    True, 'Whether to consider donor information', is_key=None)
  use_wise_man = Flag.boolean(False, 'Option to turn on wiseman', is_key=None)

  val_token = Flag.integer(None, 'Token for validation set', is_key=None)
  test_token = Flag.integer(None, 'Token for test set', is_key=None)

  loss_coef = Flag.float(1.0, 'Loss coefficient', is_key=None)
  only_BT = Flag.boolean(False, 'Only classify T/B cells', is_key=None)

  stop_grad = Flag.boolean(
    False, 'Whether to stop branch gradient', is_key=None)

  dim1 = Flag.integer(None, '...', is_key=None)
  dim2 = Flag.integer(None, '...', is_key=None)
  dim3 = Flag.integer(None, '...', is_key=None)
  dim4 = Flag.integer(None, '...', is_key=None)
  dim5 = Flag.integer(None, '...', is_key=None)
  dim6 = Flag.integer(None, '...', is_key=None)
  dim7 = Flag.integer(None, '...', is_key=None)
  dim8 = Flag.integer(None, '...', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
WBCHub.register()