import sys
sys.path.append('../../../')


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
summ_suffix = '_small'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('use_batchnorm', False)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
s.register('archi_string', '8-16-24-18')

s.register('epoch', 100)

s.register('lr', 0.0003)
s.register('batch_size', 32)
s.register('val_config', 'd-2')

s.register('use_wise_man', True, False)
s.register('image_side_length', 250, 300)

s.run(20)