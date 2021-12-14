import sys
sys.path.append('../')

from tframe.utils.script_helper import Helper
from pet.pet_configs import PetConfig


Helper.register_flags(PetConfig)
s = Helper()
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', 0)
s.register('gpu_memory_fraction', 0.3)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 9999)

s.register('early_stop', True)
s.register('patience', 10)

s.register('batch_size', 16, 128)
s.register('lr', 0.00001, 0.001)
s.register('lr_decay_method', 'cos', '-')
s.register('kernel_size', 3, 5)
s.register('use_batchnorm', s.true_and_false)

s.configure_engine(strategy='skopt', criterion='Best RMSE',
                   greater_is_better=False)
s.run(rehearsal=False)