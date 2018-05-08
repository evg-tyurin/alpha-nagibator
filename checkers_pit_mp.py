import datetime
from Arena_MP import Arena_MP
from utils import *

from checkers.CheckersGame import CheckersGame, display

"""
Use this script to play two NN agents against each other in a multi-threaded way.
The script is also useful for playing NN agent against random player.
"""
args = dotdict({
    'numGames': 100, 
    'mcts_threads': 20,
    
})
# random_args for Random Player
#random_args = dotdict({})

# args1 for NNet player #1
args1 = dotdict({
    'numMCTSSims': 50, 
    'cpuct':1.0, 
    'dirAlpha':0.3, 
    'epsilon':0,
    
    'load_model': True,
    'load_folder_file': ('/bigdata/alpha-nagibator/checkers/keras/rev.250-1/','checkpoint_?.pth.tar'),
})
# args2 for NNet player #2
args2 = dotdict({
    'numMCTSSims': 100, 
    'cpuct':1.0, 
    'dirAlpha':0.3, 
    'epsilon':0,
    
    'load_model': True,
    'load_folder_file': ('/bigdata/alpha-nagibator/checkers/keras/rev.250-1/checkpoint','checkpoint_?.pth.tar'),
})

assert args.numGames % 2 == 0, "Number of games should be a multiple of 2"



if __name__=="__main__":
    
    """ Executes the given number of episodes """
    game = CheckersGame()
    
    print("match of", args.numGames, "games in", args.mcts_threads, "threads")
    print("start at ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    #Arena_MP().play(game, args, random_args, args1)
    #Arena_MP().play(game, args, args1, args2)
    
    for n in [45,46,47,48,49]:
        args1.load_folder_file = (args1.load_folder_file[0],'checkpoint_'+str(35)+'.pth.tar')
        args2.load_folder_file = (args2.load_folder_file[0],'checkpoint_'+str(n)+'.pth.tar')
        
        # match two NN
        print("Play two NN of ", args1.load_folder_file, " vs. ", args2.load_folder_file)
        Arena_MP().play(game, args, args1, args2, None, None)
        
        # match Random vs. NN
        #print("Play RandomPlayer vs. NN of ", args1.load_folder_file)
        #Arena_MP().play(game, args, random_args, args1)

    print("end of match at ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
