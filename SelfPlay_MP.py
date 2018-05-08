from __future__ import print_function

from collections import deque
import numpy as np
import time, datetime, os, sys
from pickle import Pickler, Unpickler
from multiprocessing import Process, Queue, Manager

from MCTS import MCTS
from utils import *
from utils_mp import *
# from checkers.keras.NNet_MP import NNetWrapper as nn
from checkers.keras.NNet_Client import NNet_Client

"""
Manager for multi-threaded Self-Play phase.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 8, 2018.

"""
class SelfPlay_MP():
    """ Manager for multi-threaded Self-Play.
            nnet : instance of NNet_MP.NNetWrapper
    """
    
    def executeEpisodes(self, game, nnet, args, iteration):
        """ Executes the given number of episodes """
        self.game = game
        self.nnet = nnet
        self.args = args
        numEps = self.args.numEps
        self.episodeQueue = Queue(numEps) # all episodes that should be processed, their numbers are from 1 to numEps
        for num in range(1, numEps+1):
            self.episodeQueue.put(num)
            
        manager = Manager()
        self.activeEpisodes = manager.dict() # episodes that are currently being processed by a workers
        
        # folder for temp files of processes 
        folder = "temp/checkers_process"
        clean_folder(folder)

        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        
        numProcesses = self.args.mcts_threads
        episode = {
            # 'numEps': epsPerProcess,
            'game_module': self.game.__module__,
            'game_class': self.game.__class__.__name__,
            'args': self.args                
        }
        filename = folder+"/selfplay-"+str(iteration)+".dat"
        with open(filename, "wb+") as f:
            Pickler(f).dump(episode)
        f.closed
        
        #if not self.nnet_desc:
        #    nnet = nn(self.game)
        #else:
        #    nnet = nn(self.game)
        #    nnet.load_checkpoint(self.nnet_desc[0], self.nnet_desc[1])
            
        #episode_num_pipe_list = self.get_pipes(numProcesses)
        pipe_list = self.nnet.get_pipes(numProcesses)
        workers = []
        
        for pipe in pipe_list:
            p = Process(target=executeEpisodePlan, args=(filename, pipe, self.episodeQueue, self.activeEpisodes))
            workers.append(p)
            p.start()
            
        # Watch list of workers and check if a worker had terminated abnormally
        monitor_terminated_processes(workers, self.activeEpisodes, self.episodeQueue)
            
        # Wait for all threads to complete
        for proc in workers:
            proc.join()
            
        manager.shutdown()

        # read trainExamples from prepared files
        missingFiles = 0
        for episodeNum in range(1, numEps+1):
            examplesFile = filename+"-"+str(episodeNum)+".examples"
            if os.path.isfile(examplesFile):
                with open(examplesFile, "rb") as f:
                    examples = Unpickler(f).load()
                f.closed
                print("parallel result: episode=", episodeNum, "=>", len(examples), "example(s)")
                trainExamples += examples
            else:
                print("missing file:", examplesFile)
                missingFiles += 1
        print("missing files: ", missingFiles)
        assert missingFiles <= 1, "missingFiles="+str(missingFiles)
        
        #print "All Threads were Joined"
        assert len(trainExamples)>0, "No trainExamples collected"
        return trainExamples
    
    
def executeEpisodePlan(filename, send_end, episodeQueue, activeEpisodes):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                       pi is the MCTS informed policy vector, v is +1 if
                       the player eventually won the game, else -1.
    """
    # avoid identical games in the threads
    np.random.seed(os.getpid())
    
    with open(filename, "rb") as f:
        episodePlan = Unpickler(f).load()
    f.closed
    
    game = getattr(sys.modules[episodePlan['game_module']], episodePlan['game_class'])()
    nnet = NNet_Client(game, send_end)
    args = episodePlan['args']
    
    # working folder is Debug/Coach/[date_time]/[pid]
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+str(os.getpid())
    args.folder = os.path.join(args.folder, dt)
    os.makedirs(args.folder)
        
    while True:
        #for n in range(episodePlan['numEps']):
        start = time.time()
        #episodeNum = str(os.getpid())+"."+str(n+1)
        if not episodeQueue.empty():
            episodeNum = episodeQueue.get()
            activeEpisodes[os.getpid()] = episodeNum
        elif len(activeEpisodes)==0:
            print("[", os.getpid(), "] SelfPlay worker exits")
            break
        else:
            time.sleep(5)
            continue
        
        print("[", os.getpid(), "] SelfPlay.executeEpisode:", episodeNum)
        mcts = MCTS(game, nnet, args)   # reset search tree
        # mcts.id = "mcts-"+episodeNum
        trainExamples = executeEpisode(episodeNum, game, mcts, args)
        elapsed = time.time() - start
        print("[", os.getpid(), "] SelfPlay.executeEpisode:", episodeNum, "took", elapsed, "s")
        # persist collected examples of the episode to a file
        examplesFile = filename+"-"+str(episodeNum)+".examples" 
        with open(examplesFile, "wb+") as f:
            print("save "+examplesFile+":", len(trainExamples))
            Pickler(f).dump(trainExamples)
        f.closed
        
        # report OK to the main process
        activeEpisodes.pop(os.getpid(), None)
        
        del trainExamples[:]
    
    # send stop signal
    print("[", os.getpid(), "] send STOP to pipe")
    send_end.send((0,0))
    send_end.close()
    

def executeEpisode(episode, game, mcts, args):    
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        #print("step:", episodeStep)
        canonicalBoard = game.getCanonicalForm(board,curPlayer)
        canonicalBoard.id = "eps"+str(episode)+"_step"+str(episodeStep)+"_0"
        temp = int(episodeStep < args.tempThreshold)

        #start = millis()
        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        #time = millis() - start
        #print("[",mcts.id,"] getActionProb:", mcts.numActionProbs, "took", time, "ms") 
        sym = game.getSymmetries(canonicalBoard, pi)
        for b,p in sym:
            # stringRepr is used for preprocessing examples, not for training
            s = list(canonicalBoard.stringRepr)
            # 100?1010007001000707101000700100070710100070010007071010007001000707--00
            assert s[3] in ("r","n"), "String representation of the board was changed"
            s[3] = "?" # reset rotation flag
            s = "".join(s)
            
            trainExamples.append([b, curPlayer, p, s])

        action = np.random.choice(len(pi), p=pi)
        action = canonicalBoard.transform_action_for_board(action, board)
        board.legal_moves = canonicalBoard.legal_moves
        try:
            board, curPlayer = game.getNextState(board, curPlayer, action)
        except AssertionError:
            board.display()
            raise
        
        r = game.getGameEnded(board, curPlayer)

        if r!=0:
            game.printGameRecord(board, curPlayer, args.folder)
            return [(x[0], x[2], r*((-1)**(x[1]!=curPlayer)), x[3]) for x in trainExamples]
