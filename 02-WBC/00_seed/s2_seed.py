import sys
sys.path.append('../../')


from tframe.utils.script_helper import Helper
from wbc.wbc_hub import WBCHub


s = Helper()
s.register_flags(WBCHub)
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
summ_suffix = '_s02'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('epoch', 100)
s.register('archi_string',
           '12-16-24-32=32-16',
           '32-64-32-16=64-16',
           '64-48-32-24=32-16',
           '64-48-32-24=80-32')
s.register('strides', 3)
s.register('activation', 'tanh', 'relu')
s.register('lr', 0.003, 0.0003)
s.register('use_wise_man', True, False)

s.run(100)