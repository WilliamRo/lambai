import sys
sys.path.append('../../')


from tframe.utils.script_helper import Helper
from wbc.wbc_hub import WBCHub


s = Helper()
s.register_flags(WBCHub)
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
summ_suffix = '_0'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.75)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('epoch', 1000)
s.register('test_config', 'c-!r-100')
s.register('val_config', 'c-!r-100')

# s.register('archi_string', '12-16-24-32=32-16',)
s.register('dim1', 12, 16, 32, hp_type=int)
s.register('dim2', 8, 16, 32, hp_type=int)
s.register('dim3', 16, 24, 32, hp_type=int)
s.register('dim4', 16, 32, hp_type=int)

s.register('dim5', 16, 32, 64, hp_type=int)
s.register('dim6', 16, 32, 64, hp_type=int)

s.register('strides', 2, 3, hp_type=int)
s.register('kernel_size', 3, 4, 5, hp_type=int)
s.register('image_side_length', 300, 250)
s.register('activation', 'relu')
s.register('lr', 0.0003, 0.001, 0.003)

s.register('use_wise_man', True, False)
s.register('use_batchnorm', True)
s.register('patience', 8, 12, 20, hp_type=int)
s.register('batch_size', 20, 100)

s.set_hp_property('lr', hp_type=float, scale='log-uniform')
s.set_hp_property('batch_size', hp_type=int)

s.configure_engine(strategy='skopt', criterion='Test F1')
s.configure_engine(auto_set_hp_properties=False)
s.run(rehearsal=False)