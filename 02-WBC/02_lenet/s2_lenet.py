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
summ_suffix = '_s00'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('epoch', 100)
s.register('archi_string',
           '64-56-48-32-32',
           '32-32-32-32-32',
           '64-64-32-32',
           '32-32-32-32',
           '64-32-32',
           '32-32-32')
s.register('lr', 0.0003, 0.003)
s.register('batch_size', 32, 64)

s.run(100)