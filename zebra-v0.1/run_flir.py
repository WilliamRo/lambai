try:
  from lambo.zebra.gui.zinci import Zinci
  from lambo.zebra.io.flir import FLIRFetcher
  from lambo.zebra.decoder.subtracter import Subtracter

  z = Zinci(height=6, width=6, max_fps=50)
  z.set_fetcher(FLIRFetcher())
  z.set_decoder(Subtracter(boosted=True))

  z.display()
except Exception as e:
  print(e)
  input('Press any key to quit ...')
