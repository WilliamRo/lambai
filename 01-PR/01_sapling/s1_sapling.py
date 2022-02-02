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
s.register('gpu_memory_fraction', 0.8)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
# s.register('feature_type', 9)

s.register('archi_string', '32-32-24-16-8', '32-32-32-24-24-16')
s.register('nap_token', 'beta')
s.register('batch_size', 16)
# s.register('lr', 0.0001)
s.register('loss_string', 'BER')
s.register('data_token', 'eta')
s.register('train_config', '2a3')
# s.register('alpha', 0.0, 0.5, 1.0)
# s.register('epoch', 200)
# s.register('kernel_size', 10)
# s.register('dilations', 8)
# s.register('global_activation', 'relu', 'sigmoid')
# s.register('input_projection', s.true_and_false)
s.register('input_projection', True)

# s.configure_engine(greater_is_better=False)
# s.run(rehearsal=True)
s.run(times=1, add_script_suffix=True)
