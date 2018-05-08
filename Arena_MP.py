from __future__ import print_function

from pickle import Pickler, Unpickler
from multiprocessing import Process, Queue, Manager
import time, os, sys

from Arena import Arena
from utils import clean_folder
from utils_mp import *
from checkers.CheckersPlayers import *
from checkers.keras.NNet_MP import NNetWrapper as NNet
from checkers.keras.NNet_Client import NNet_Client

import numpy as np

from checkers.CheckersGame import CheckersGame, display

"""
Multiprocessing implementation of Arena.
Let you run multiple competitions in parallel.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 8, 2018.

"""
class Arena_MP():
    
    def play(self, game, args, args1, args2, referenceNetwork=None, testedNetwork=None):
        """ Play a number of episodes specified in args
                args: settings for the competition
                args1: either settings for NN player1 or empty dict for random player
                args2: settings for NN player2 
        """
        # folder for temp files of processes 
        folder = "temp/checkers_process"
        clean_folder(folder)

        numEps = int(args.numGames/2)
        self.episodeQueue = Queue(numEps) # all 2-games episodes that should be processed, their numbers are from 1 to numEps
        for num in range(1, numEps+1):
            self.episodeQueue.put(num)
            
        manager = Manager()
        self.activeEpisodes = manager.dict() # episodes that are currently being processed by a workers
        self.waitingWorkers = manager.dict() # workers that are waiting for new work        
         
        #episodePlans = []
        numProcesses = args.mcts_threads
        #for num in range(numProcesses):
        #epsPerProcess = int(numEps/numProcesses)
        episode = {
            #'numEps': epsPerProcess,
            'game_module': game.__module__,
            'game_class': game.__class__.__name__,
            'args': args,                
            'args1': args1,                
            'args2': args2,                
        }
        filename = folder+"/Arena.dat"
        with open(filename, "wb+") as f:
            Pickler(f).dump(episode)
        f.closed
        #episodePlans.append(filename)
        
        if referenceNetwork:
            nnet1 = referenceNetwork
        else:
            if args1: # else this is random_player
                nnet1 = NNet(game)
                nnet1.load_checkpoint(args1.load_folder_file[0], args1.load_folder_file[1])
            
        if testedNetwork:
            nnet2 = testedNetwork
        else:
            nnet2 = NNet(game)
            nnet2.load_checkpoint(args2.load_folder_file[0], args2.load_folder_file[1])
        
        if args1 or testedNetwork:
            pipe_list_1 = nnet1.get_pipes(numProcesses)
        else:
            pipe_list_1 = [None]*numProcesses
            
        pipe_list_2 = nnet2.get_pipes(numProcesses)
        workers = []
        
        for (pipe1, pipe2) in zip(pipe_list_1, pipe_list_2):
            p = Process(target=executeEpisodePlan, args=(filename, pipe1, pipe2, self.episodeQueue, self.activeEpisodes, self.waitingWorkers))
            workers.append(p)
            p.start()
            
        # Watch list of workers and check if a worker had terminated abnormally
        monitor_terminated_processes(workers, self.activeEpisodes, self.episodeQueue)
            
        # Wait for all threads to complete
        for proc in workers:
            proc.join()
    
        manager.shutdown()

        # read results from prepared files
        missingFiles = 0
        totalWin, totalLost, totalDraw = 0, 0, 0
        for episodeNum in range(1, numEps+1):
            examplesFile = filename+"-"+str(episodeNum)+".examples"
            if os.path.isfile(examplesFile):
                with open(examplesFile, "rb") as f:
                    (win, lost, draw) = Unpickler(f).load()
                f.closed
                print("parallel result, win/lost/draw: ", (win, lost, draw))
                totalWin += win
                totalLost += lost
                totalDraw += draw
            else:
                print("missing file:", examplesFile)
                missingFiles += 1
        
        info = ""
        if missingFiles>0:
            info = "missingFiles: "+str(missingFiles)
        print("Total result, win/lost/draw: ", (totalWin, totalLost, totalDraw), info)
        #print "All Threads were Joined"
        #assert len(trainExamples)>0, "No trainExamples collected"
        print("END")

    
""" The following methods are invoked by spawn processes """ 

def executeEpisodePlan(filename, send_end_1, send_end_2, episodeQueue, activeEpisodes, waitingWorkers):
    """
    This function executes one episode of Arena, starting with player 1.
    As the game is played, results are saved to a file. The game is played till the game ends. 

    It uses a temp=0.
    
    send_end_1: either Pipe for NN player1 or None for random player
    send_end_2: Pipe for NN player2

    Returns:
        nothing, all results are saved in a file.
    """
    # avoid identical games in the threads
    #print("set seed to ", os.getpid())
    np.random.seed(os.getpid())
    
    with open(filename, "rb") as f:
        episodePlan = Unpickler(f).load()
    f.closed
    
    game = getattr(sys.modules[episodePlan['game_module']], episodePlan['game_class'])()

    args1 = episodePlan['args1']
    args2 = episodePlan['args2']
        
    if send_end_1:
        nnet1 = NNet_Client(game, send_end_1)
        n1p = NNetPlayer(game, nnet1, args1)
    else:
        n1p = RandomPlayer(game)
    
    nnet2 = NNet_Client(game, send_end_2)
    n2p = NNetPlayer(game, nnet2, args2)

    while True:
        #for n in range(episodePlan['numEps']):
        start = time.time()
        #episodeNum = str(os.getpid())+"."+str(n+1)
        if not episodeQueue.empty():
            episodeNum = episodeQueue.get()
            activeEpisodes[os.getpid()] = episodeNum
            waitingWorkers.pop(os.getpid(), None)
        elif len(activeEpisodes)==0:
            print("[", os.getpid(), "] SelfPlay worker exits")
            waitingWorkers.pop(os.getpid(), None)
            break
        elif len(waitingWorkers)>2:
            print("[", os.getpid(), "] SelfPlay worker exits due to a large number of waiting workers")
            waitingWorkers.pop(os.getpid(), None)
            break
        else:
            waitingWorkers[os.getpid()] = 1
            time.sleep(5)
            continue
        
        start = time.time()
        #episodeNum = str(os.getpid())+"."+str(n+1)
        # basic episode always consists of two games, each player plays white and black
        numGames = 2
        print("[",os.getpid(),"] Arena: play", numGames, "games")
        
        arena = Arena(n1p, n2p, game, display=display)
        (win, lost, draw) = arena.playGames(numGames, verbose=True)
        print("[",os.getpid(),"] Arena: win/lost/draw", (win, lost, draw))
        
        elapsed = time.time() - start
        print("[",os.getpid(),"] Arena total time: ", numGames, "games took", elapsed, "s")
        
        # persist collected results to a file
        examplesFile = filename+"-"+str(episodeNum)+".examples" 
        with open(examplesFile, "wb+") as f:
            print("save "+examplesFile+":", (win, lost, draw))
            Pickler(f).dump((win, lost, draw))
        f.closed
        
        # report OK to the main process
        activeEpisodes.pop(os.getpid(), None)
    
    # send stop signal
    print("[", os.getpid(), "] send STOP to pipe(s)")
    if send_end_1:
        send_end_1.send((0,0))
        send_end_1.close()
    send_end_2.send((0,0))
    send_end_2.close()
    
