from __future__ import print_function

import os, sys, pprint
from utils import *
from checkers.CheckersGame import CheckersGame as Game

from Coach import Coach as Coach

# single-threaded self-play
# from checkers.keras.NNet import NNetWrapper as nn
# from SelfPlay import SelfPlay as SelfPlay

# multi-threaded self-play
from checkers.keras.NNet_MP import NNetWrapper as nn
from SelfPlay_MP import SelfPlay_MP as SelfPlay

work_folder = '/bigdata/alpha-nagibator/checkers/keras/rev.250-1/'

args = dotdict({
    'numIters': 1,
    'numEps': 200,
    'numMCTSSims': 100,

    'tempThreshold': 15,
    'maxlenOfQueue': 250000,
    'updateThreshold': 0.60, # not used if skipArena=True
    'arenaCompare': 40, # not used if skipArena=True
    'cpuct': 1,
    'dirAlpha': 0.3,
    'epsilon': 0.25, 

    'checkpoint': work_folder+'checkpoint/',
    'load_model': True,
    'load_folder_file': (work_folder,'?'),
    #'loadExamplesFromSameIteration': False, # same or previous 
    'next_iteration': '?',
    
    'skipArena': True,
    'maxCheckpointFiles': 15,
    'saveCheckpointEvery': 1,
    'numItersForTrainExamplesHistory': 20,
    
    'mcts_threads': 20, # used by SelfPlay_MP only
    'pipe_timeout': 0.005, # overrides setting in NNet.args, // 0.01 and 0.004 were tested and is OK for 20 threads of self-play
})

if __name__=="__main__":
    
    # overwrite next_iteration if it was specified via command line argument
    if len(sys.argv)==2:
        print("set args.next_iteration =", sys.argv[1])
        args.next_iteration = int(sys.argv[1])
    
    if os.path.isfile("stop"):
        os.rename("stop","=stop")

    g = Game()
    nnet = nn(g)
    
    print("main.args = ", end="")
    pprint.pprint(args, indent=4)
    print("nnet.args = ", end="")
    pprint.pprint(nnet.args, indent=4)
    
    if args.mcts_threads and not args.pipe_timeout:
        assert args.pipe_timeout, "args.pipe_timeout is not set"
        
    nnet.args.pipe_timeout = args.pipe_timeout

    c = Coach(g, nnet, args, SelfPlay())
    c.learn()
