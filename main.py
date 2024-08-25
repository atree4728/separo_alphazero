import arena
import separo
import numpy as np
import players
import ray


width = 9
# blue = players.NaiveMCPlayer(separo.Color.Blue, 2)
red = players.PUCTMCPlayer(separo.Color.Red, 2, np.sqrt(2), 5, width)
blue = players.PUCTMCPlayer(separo.Color.Blue, 2, np.sqrt(2), 5, width)
arena = arena.Arena(width, red, blue)
ray.init()
print(arena.play_matchs(100))
