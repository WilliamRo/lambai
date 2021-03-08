import sys
sys.path.append('../')

from cf10_core import th
from tframe.utils.script_helper import Helper


s = Helper('../t81_lenet.py')
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = '0306_s81_lenet'
summ_suffix = '__dev'
gpu_id = 0

s.register('gather_summ_name', summ_name + summ_suffix + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.3)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 2)
s.register('lr', 0.003, 0.0003, hp_type=float, scale='log')
s.register('kernel_size', 2, 4, hp_type=int)
s.register('augmentation', s.true_and_false)


rehearsal = True
strategy = ['grid', 'skopt'][1 if rehearsal else 0]

s.configure_engine(strategy=strategy, criterion='Test accuracy',
                   expectation=72)
s.run(rehearsal=rehearsal, times=50)