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
summ_suffix = '_grad_v2'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
# s.register('lr', 0.003, 0.0003)
# s.register('batch_size', 32)
# s.register('augmentation', False, True)
# s.register('loss_coef', 1.0, 0.0)
only_BT = False
s.register('only_BT', only_BT)

s.register('loss_coef', 1.0)
s.register('use_wise_man', True)
s.register('patience', 5)
s.register('stop_grad', True, False)

# s.register('suffix', 'BT' if only_BT else 'BTGM')

s.run(50)