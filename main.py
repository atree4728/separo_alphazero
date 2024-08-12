import arena
import separo
import numpy as np
import players
import ray


width = 9
rng = np.random.default_rng()
red = players.NaiveMCPlayer(separo.Color.Red, 1, rng)
blue = players.PUCTMCPlayer(separo.Color.Blue, 1, rng, np.sqrt(2), 5, width)
# print(arena.play(width, red, blue, True))
arena = arena.Arena(width, red, blue)
ray.init()
print(arena.play_matchs())
