from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class PetConfig(SmartTrainerHub):

  class PetKeys(object):
    pass


  pet_input_size = Flag.integer(None, 'Input image size', is_key=None)
  use_pet_image = Flag.boolean(True, 'Whether to use pet image', is_key=None)
  pet_mean = Flag.float(-1, 'Set to [1, 100] to use the conservative strategy',
                        is_key=None)
  pet_radius = Flag.float(5.0, 'Radius used in conservative strategy',
                          is_key=None)

  meta_branch_code = Flag.string(None, 'Architecture string of meta branch',
                                 is_key=None)
  by_pass_image = Flag.boolean(False, 'Whether to bypass image', is_key=None)


  @property
  def pet_input_shape(self):
    L = self.pet_input_size
    # assert isinstance(L, int)
    return [L, L, 3]

  @property
  def use_meta_data(self):
    return not self.meta_branch_code in (None, '-', 'x')

  @property
  def use_classifier(self):
    return self.num_classes not in (-1, None)

  def check_num_classes(self):
    assert self.num_classes in (25, 50, 100)


# New hub class inherited from SmartTrainerHub must be registered
PetConfig.register()