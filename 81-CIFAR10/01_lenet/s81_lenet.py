import sys
sys.path.append('../')

from cf10_core import th
from tframe.utils.script_helper import Helper


s = Helper()
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name

summ_suffix = '_00'
ckpt_suffix = '_0'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.3)
s.register('suffix', ckpt_suffix)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('activation', 'relu')

s.register('lr', 0.003, 0.001, 0.0003)
s.register('batch_size', 16, 32, 64, 128)
s.register('kernel_size', 3, 4, 5)
s.register('patience', 5, 10)
s.register('augmentation', s.true_and_false)

s.configure_engine(strategy='skopt', criterion='Test accuracy')
s.run(rehearsal=False)