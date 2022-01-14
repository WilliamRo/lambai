try:
  import os
  from lambo.zebra.gui.zinci import Zinci
  from lambo.zebra.io.pseudo import PseudoFetcher
  from lambo.zebra.decoder.subtracter import Subtracter

  trial_root = r'E:\lambai\01-PR\data'
  trial_names = ['01-3t3', '80-spacer-0526']
  path = os.path.join(trial_root, trial_names[1])

  z = Zinci(height=6, width=6, max_fps=50)
  z.set_fetcher(PseudoFetcher(path, fps=20, L=1000, seq_id=4))
  z.set_decoder(Subtracter(boosted=True))

  z.display()
except Exception as e:
  print(e)
  input('Press any key to quit ...')

