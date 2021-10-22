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
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
s.register('gpu_memory_fraction', 0.4)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)

s.register('num_stacks', 2)
s.register('module_per_stack', 2)
s.register('input_projection', False)
s.register('developer_code', 'mg')

# s.register('developer_code', 'mg', 'dc', '-')

s.configure_engine(times=100)
s.run(rehearsal=False)