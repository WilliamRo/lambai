import sys
sys.path.append('../../')


from tframe.utils.script_helper import Helper
from wbc.wbc_hub import WBCHub


s = Helper()
s.register_flags(WBCHub)
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
s.register('epoch', 100)

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
suffix = '_s00'
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('lr', 0.001, 0.0001)
s.register('batch_size', 32, 64)
s.register('dropout', 0.2, 0.4)

s.run(2)
