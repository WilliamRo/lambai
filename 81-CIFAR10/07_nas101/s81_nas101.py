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

summ_suffix = '_01'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', True)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)

# s.register('bn_momentum', 0.99, 0.997, 0.999)
# s.register('bn_epsilon', 1e-3, 1e-4, 1e-5)
# s.register('patience', 10)
s.register('developer_code', 'mg', 'dc', '-')

# s.register('lr', 0.003, 0.001, 0.0001, hp_type=float)
# s.register('batch_size', 32, 64, 128, hp_type=int)
# s.register('filters', 32, 64, 96, hp_type=list)
# s.register('num_stacks', 1, 2, 3, hp_type=int)
# s.register('module_per_stack', 1, 2, 3, hp_type=int)
# s.register('patience', 5, 10, hp_type=int)

# s.configure_engine(strategy='skopt', criterion='Test accuracy')
s.configure_engine(times=100)
s.run(rehearsal=False)