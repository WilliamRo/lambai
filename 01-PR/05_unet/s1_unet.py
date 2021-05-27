import sys
sys.path.append('../../')


from tframe.utils.script_helper import Helper
from pr.pr_configs import PRConfig


s = Helper()
s.register_flags(PRConfig)
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
summ_suffix = '_t4_rev'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.75)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)

s.register('int_para_1', *list(range(10, 15)))
s.configure_engine(times=1)
s.run(rehearsal=False)