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
ckpt_suffix = '_t00'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.3)
# s.register('prefix', '0323_')
s.register('suffix', ckpt_suffix)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('overwrite', False)
s.register('early_stop', True)
s.register('patience', 20)

s.register('batch_size', 16, 256)
s.register('lr', 0.00001, 0.01)
s.register('lr_decay_method', 'cos', '-')
s.register('dropout', 0.05, 0.6)
s.register('decoupled_l2_penalty', 0.0, 0.001)

# s.constrain({'lr_decay_method': 'cos'}, {'decoupled_l2_penalty': 0})

s.configure_engine(strategy='skopt', criterion='Improvement')
# s.configure_engine(times=10)
s.run(rehearsal=False)