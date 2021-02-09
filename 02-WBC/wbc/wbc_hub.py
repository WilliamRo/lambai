from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class WBCHub(SmartTrainerHub):

  image_height = Flag.integer(350, 'Blood cell image height', is_key=None)
  image_width = Flag.integer(320, 'Blood cell image width', is_key=None)

  with_donor = Flag.boolean(
    True, 'Whether to consider donor information', is_key=None)

  val_token = Flag.integer(None, 'Token for validation set', is_key=None)
  test_token = Flag.integer(None, 'Token for test set', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
WBCHub.register()