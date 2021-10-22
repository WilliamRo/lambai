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
summ_suffix = '_BT'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('epoch', 100)
# s.register('archi_string',
#            '12-24-32=64-32',
#            '6-16-24-32=64-32',
#            '6-16-24=64-32',
#            )

s.register('strides', 2, 3)
s.register('lr', 0.0003, 0.0001)
s.register('batch_size', 32, 64 ,128)
s.register('augmentation', True)
s.register('only_BT', True, False)

s.constrain({'lr': 0.0001}, {'batch_size': (64, 128)})

s.run(20, rehearsal=True)