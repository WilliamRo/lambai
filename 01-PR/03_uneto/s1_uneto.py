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
s.register('allow_growth', True)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('rehearse', False)
s.register('win_size', 512)
s.register('batch_size', 16, 24)
s.register('loss_string', 'gbe')

s.register('data_token', 'eta')
s.register('train_config', '2a3', '4a5')

s.run(times=1, add_script_suffix=True)
