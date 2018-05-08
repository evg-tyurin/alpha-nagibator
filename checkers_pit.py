import Arena
from MCTS import MCTS
from checkers.CheckersGame import CheckersGame, display
from checkers.keras.NNet import NNetWrapper as NNet
from checkers.CheckersPlayers import *

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = CheckersGame()

# all players
#rp = RandomPlayer(g)
hp = HumanCheckersPlayer(g)
#rp2 = RandomPlayer(g)
# ep = EngineCheckersPlayer(g,"./checkers/engine/KestoG_1_4_Moscow.dll")
# http = HttpCheckersPlayer(g,"http://10.168.1.101:8989/")
                          #"http://127.0.0.1:8989/")

# nnet players
args1 = dotdict({
    'numMCTSSims': 100, 
    'cpuct':1.0, 
    'dirAlpha':0.3, 
    'epsilon':0,
    
    'load_model': True,
    'load_folder_file': ('pretrained_models/checkers/keras','checkpoint_51_eps400_mcts100.pth.tar'),
})
n1 = NNet(g)
n1.load_checkpoint(args1.load_folder_file[0],args1.load_folder_file[1])
n1p = NNetPlayer(g, n1, args1)


#n2 = NNet(g)
#n2.load_checkpoint('../models-for-alpha/checkers/keras/rev.231','checkpoint_2.pth.tar')
#args2 = dotdict({'numMCTSSims': 100, 'cpuct':1.0, 'dirAlpha':0.3, 'epsilon':0})
#n2p = NNetPlayer(g, n2, args2)

for n in range(1):    
    arena = Arena.Arena(hp, n1p, g, display=display)
    print(n+1, ": win/lost/draw", arena.playGames(20, verbose=True))
