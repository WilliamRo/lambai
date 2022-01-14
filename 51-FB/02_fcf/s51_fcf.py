import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')


from tframe.utils.script_helper import Helper
from fb.fb_config import FBConfig


Helper.register_flags(FBConfig)
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
s.register('gpu_memory_fraction', 0.3)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('patience', 100)
s.register('visualize_after_training', False)

s.register('lr', 0.00003, 0.003)
s.register('batch_size', 8, 16, 32, 64)

s.register('kernel_size', 3, 5)
s.register('filters', 16, 32, 64)
s.register('floor_height', 1, 2, 4)
s.register('use_batchnorm', s.true_and_false)

s.register('auto_bound', s.true_and_false)
s.register('yolo_noob', 0.01, 0.5)


s.configure_engine(strategy='skopt', criterion='Best APC')
s.configure_engine(greater_is_better=True)
s.run()
