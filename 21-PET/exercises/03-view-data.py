"""
Run this script directly to see what happens.
"""
import sys
sys.path.insert(0, '../')  # TODO: uncomment to switch to main branch

from pet_core import th
from pet.pet_agent import PetAgent


th.pet_input_size = 256  # TODO: Uncomment this line to see what happens
ps = PetAgent.load_as_tframe_data(th.data_dir, traverse=True)
ps.visualize()

