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
summ_suffix = '_00'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', True)
s.register('gpu_memory_fraction', 0.75)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)

s.register('win_size', 360)
s.register('patience', 50)
s.register('num_layers', 6, 5, 4, 3)
s.register('filters', 56, 40, 24)
s.register('kernel_size', 5, 4)

s.configure_engine(times=5)
s.run(rehearsal=False)