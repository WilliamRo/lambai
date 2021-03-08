import sys
sys.path.append('../../')


from tframe.utils.script_helper import Helper
from wbc.wbc_hub import WBCHub


s = Helper()
s.register_flags(WBCHub)
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
s.register('epoch', 1)

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
suffix = '_s00'
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('archi_string', '1-1')
s.register('lr', 0.003, 0.0003)
s.register('batch_size', 32, 50, 64)
s.constrain({'lr': 0.003}, {'batch_size': (32, 50)})
s.constrain({'lr': 0.0003}, {'batch_size': 64})

s.run(10)
