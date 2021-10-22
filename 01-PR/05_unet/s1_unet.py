import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')


from tframe.utils.script_helper import Helper
from pr.pr_configs import PRConfig


Helper.register_flags(PRConfig)
s = Helper()
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
summ_suffix = ''
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.75)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
# s.register('feature_type', 9)

# s.register('batch_size', 4, 32)
# s.register('lr', 0.0001, 0.003)
# s.register('filters', 8, 32)
# s.register('contraction_kernel_size', 3, 9)
# s.register('half_height', 3, 4)
s.register('bridges', '-', 'a', '1,2,3', '4')

# s.configure_engine(strategy='skopt', criterion='Test WMAE')
# s.configure_engine(greater_is_better=False)
s.run()
