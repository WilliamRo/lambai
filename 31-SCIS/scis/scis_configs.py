from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class SCISConfig(SmartTrainerHub):

  cell_type = Flag.string(None, 'Should be in (c, a, s)', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
SCISConfig.register()