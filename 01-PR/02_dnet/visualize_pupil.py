from pr.architectures.konjac import plot_dual
from tframe.utils.note import Note

import os


path = r'./misc.dict'
# path = r'\\172.16.233.191\wmshare\projects\lambai\01-PR\02_dnet\checkpoints\0707_dnet(L-25-omega-5-rs-0.5)_02\misc.dict'

assert os.path.exists(path)

# Load note
misc = Note.load(path)

STEP_KEY, PACKAGE_KEY = 'STEP', 'PACKAGE'
plot_dual(misc[STEP_KEY], misc[PACKAGE_KEY])
