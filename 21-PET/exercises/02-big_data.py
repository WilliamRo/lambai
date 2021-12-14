"""
For this exercise you are going to wrap your data into tframe.DataSet format.
You need to finish pet_agent.py and pet_set.py
Note that it is not wise to load the pet images all at once since they will take
around 20GB of your RAM.
So the goal of this exercise is to make sure the memory usage is below some
threshold, say, 2GB, during training.

Run this script directly to see whether you have done this right.
Make sure that you have installed psutil package before running.
Note that no modification should be made to this script.
"""
import sys
# sys.path.insert(0, '../')  # TODO: uncomment to switch to main branch
import numpy as np
import psutil

from tframe import console

from pet_core import th
from pet.pet_agent import PetAgent
from pet.pet_set import PetSet


def get_memory_stats():
  stats = psutil.virtual_memory()
  return [v / (1024 ** 3) for v in (stats.used, stats.total)]

Hs, Ws = [], []

def update_model(batch):
  assert isinstance(batch, PetSet)
  for im in batch.features:
    assert isinstance(im, np.ndarray)
    Hs.append(im.shape[0])
    Ws.append(im.shape[1])


# Load dataset
ps = PetAgent.load_as_tframe_data(th.data_dir)

# Set the threshold
threshold = 2

# Pretend that we are training
success = True
log = []
batch_size = 100
for i, batch in enumerate(
    ps.gen_batches(batch_size, shuffle=True, is_training=True)):

  # Do something with this data batch
  update_model(batch)

  # Check memory usage
  used, total = get_memory_stats()
  log.append(used)
  usage = max(log) - min(log)
  console.show_status(f'Memory usage: {usage:.2f} GB')
  if usage > threshold:
    success = False
    break

  # Print progress
  console.print_progress(i, ps.size // batch_size)

if success:
  console.show_status('Congratulations!')
  console.show_info(f'H range: [{min(Hs)}, {max(Hs)}],'
                    f' W range: [{min(Ws)}, {max(Ws)}]')
else: console.show_status('Mission Failed.')
