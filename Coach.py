from __future__ import print_function

from collections import deque
import numpy as np
import time, datetime, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

from MCTS import MCTS
from Arena_MP import Arena_MP
from Arena import Arena
from utils import *
from utils_examples_global_avg import *
# from utils_examples_max_plus_overwrite import *


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args, selfPlay):
        self.game = game
        self.nnet = nnet
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.referenceNetwork = None  # the competitor network
        self.args = args
        self.selfPlay = selfPlay
        #self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+str(os.getpid())
        self.args.folder = os.path.join("Debug", "Coach", dt)
        os.makedirs(self.args.folder)
        
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        
            @param iteration: number of iteration to keep correct numbering
        """

        self.loss_value = float('inf') # loss value after the last iteration
        
        iteration = self.args.next_iteration if self.args.load_model else 1
        
        # load model and check if examples are ready
        if self.args.load_model:
            modelIteration = self.args.next_iteration - 1 
            if os.path.isfile(os.path.join(self.args.load_folder_file[0], getCheckpointFile(modelIteration))):
                self.nnet.load_checkpoint(self.args.load_folder_file[0], getCheckpointFile(modelIteration))
            else:
                self.nnet.load_checkpoint(self.args.checkpoint, getCheckpointFile(modelIteration))
            # check examples file
            examplesFile = os.path.join(self.args.load_folder_file[0], getCheckpointFile(modelIteration)+".examples.mini")
            if os.path.isfile(examplesFile):
                print("skip first Self-Play because examples file exists:", examplesFile)
                self.skipFirstSelfPlay = True
            else:
                examplesFile = os.path.join(self.args.checkpoint, getCheckpointFile(modelIteration)+".examples.mini")
                if os.path.isfile(examplesFile):
                    print("skip first Self-Play because examples file exists:", examplesFile)
                    self.skipFirstSelfPlay = True
                else:
                    self.skipFirstSelfPlay = False

        for i in range(iteration, iteration+self.args.numIters):
            # bookkeeping
            print('------ITER ' + str(i) + '------' + ', start at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>iteration:
                print('Self-Play of ITER ' + str(i))
                start = time.time()

                iterationTrainExamples = self.selfPlay.executeEpisodes(self.game, self.nnet, self.args, i)
                
                elapsed = time.time() - start
                print("all episodes took ", elapsed, "s")
                print("total examples: ", len(iterationTrainExamples))
            
                # backup examples to a file
                # NB! the examples were collected using the model from the previous iteration, so (i-1)  
                self.saveTrainExamples(i-1, iterationTrainExamples)
                
            #ask_for_continue("\nSelf play finished, continue? [y|n]\n")

            if self.args.skipArena:
                print('Optimize of ITER ' + str(i) + ', start at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                self.optimize(i)
                self.compare_networks(i)
            else:
                print('Optimize_and_Evaluate of ITER ' + str(i))
                self.optimize_and_evaluate(i)

            check_stop_condition()
        # final competition
        # ...
        
    def optimize(self, iteration):
        print("skip Arena")
        
        checkpointFile = getCheckpointFile(iteration) # "epoch.{epoch:02d}-"+
        checkpointFilePath = os.path.join(self.args.checkpoint, checkpointFile)
        
        if os.path.isfile(checkpointFilePath):
            print("Model was already trained. Skip optmizing.")
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=checkpointFile)
            return

        # shuffle examlpes before training
        trainExamplesHistory = self.loadTrainExamples(iteration-1)
        trainExamples = build_unique_examples(trainExamplesHistory)
        shuffle(trainExamples)
        shrinkToBeMultiplierOf(trainExamples, self.nnet.args.batch_size)
        print("len(examples)=", len(trainExamples))

        history = self.nnet.train(trainExamples, checkpointFilePath)
        print("=== HISTORY === \n", history.history, "\n === END OF HISTORY ===")
        
        if os.path.isfile(os.path.join(self.args.checkpoint, checkpointFile)): 
            # reload NN from the bect epoch
            print("reload model from the best checkpoint")
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename=checkpointFile)
        else:
            # checkpoint could not be automatically saved if val_loss or callback were undefined
            print("save model to checkpoint file")
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=checkpointFile)
        
        #if i % self.args.saveCheckpointEvery == 0:
        #    print("save_checkpoint for i="+str(i))
        #    self.ensureMaxCheckpoints()
        #    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=checkpointFile)
        
        # check loss in history, stop training if loss is extremely high
        max_loss_value = max(history.history['loss'])
        if max_loss_value / self.loss_value > 5:
            print("max_loss_value on the iteration is too high: ", max_loss_value, " > ", self.loss_value)
            sys.exit()
        self.loss_value = history.history['loss'][-1]
        
        trainExamples.clear() # this is deque
        del trainExamplesHistory[:] # this is list
        
    def compare_networks(self, iteration):
        arena_args = dotdict({
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
            'load_folder_file': ('/bigdata/models-for-alpha/checkers/keras/rev.250-1/batch128','checkpoint_35.pth.tar'),
        })
        # args2 for NNet player #2
        args2 = dotdict({
            'numMCTSSims': self.args.numMCTSSims, 
            'cpuct':1.0, 
            'dirAlpha':0.3, 
            'epsilon':0,

            'load_model': True,
            'load_folder_file': (self.args.checkpoint, getCheckpointFile(iteration)),
        })

        assert arena_args.numGames % 2 == 0, "Number of games should be a multiple of 2"

        """ Executes the given number of episodes """

        print("match of", arena_args.numGames, "games in", arena_args.mcts_threads, "threads")
        print("start at ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        #args1.load_folder_file = (args1.load_folder_file[0],'checkpoint_'+str(20)+'.pth.tar')
        #args2.load_folder_file = (args2.load_folder_file[0],'checkpoint_'+str(iteration)+'.pth.tar')

        # match two NN
        if not self.referenceNetwork:
            self.referenceNetwork = self.nnet.__class__(self.game)  # the competitor network
            self.referenceNetwork.load_checkpoint(args1.load_folder_file[0], args1.load_folder_file[1])
        print("Play two NN of ", args1.load_folder_file, " vs. ", args2.load_folder_file)
        Arena_MP().play(self.game, arena_args, args1, args2, self.referenceNetwork, self.nnet)

        # match Random vs. NN
        #print("Play RandomPlayer vs. NN of ", args1.load_folder_file)
        #Arena_MP().play(self.game, arena_args, random_args, args1)

        print("end of match at ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        
    def optimize_and_evaluate(self):
        # training new network, keeping a copy of the old one
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        pmcts = MCTS(self.game, self.pnet, self.args)
    
        self.nnet.train(trainExamples)
        nmcts = MCTS(self.game, self.nnet, self.args)

        # if self.args.arenaCompare is large enough both nmcts and pmcts consume huge amount of RAM.
        # RAM consumed depends also on your game implementation, particularly on game.getActionSize
        # and size of game.stringRepresentation.
        print('PITTING AGAINST PREVIOUS VERSION')
        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

        # input("Arena finished, continue?\n")

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
            print('REJECTING NEW MODEL')
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')    
        else:
            print('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
        
        print("")
        print("Previous NN MCTS stats")
        pmcts.print_stats()                
        pmcts = None
        print("New NN MCTS stats")
        nmcts.print_stats()                
        nmcts = None
        
            
    def saveTrainExamples(self, iteration, examples):
        """ Persists examples of the given iteration to a file """
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # save the last iteration examples to MINI file
        filename = os.path.join(folder, getCheckpointFile(iteration)+".examples.mini")
        with open(filename, "wb+") as f:
            # save the last iteration examples only
            Pickler(f).dump(examples)
        f.closed
        
    def loadModelAndExamples(self):
        """ Loads model and trainExamples from files
            Examples can be loaded either from the same itertaion as model or from previous one. 
            @param args.loadExamplesFromSameIteration: if True then examplesFile = modelFile[next_iteration-1] and SkipFirstSelfPlay=True 
                                                      if False then examplesFile = modelFile[next_iteration-2] and SkipFirstSelfPlay=False
        """

        if not self.args.load_model:
            ask_for_continue("args.load_model is disabled. Continue without loading model and examples? [y|n]")

        if self.args.load_model:
            modelIteration = self.args.next_iteration - 1 

            #self.args.load_folder_file = (self.args.load_folder_file[0], getCheckpointFile(modelIteration))
            self.nnet.load_checkpoint(self.args.load_folder_file[0], getCheckpointFile(modelIteration))
            
            # load examples
            if args.loadExamplesFromSameIteration:
                examplesIteration = modelIteration
                # examples based on the same model
                skipFirstSelfPlay = True
            else:
                examplesIteration = modelIteration - 1
                # examples based on the previous model
                skipFirstSelfPlay = False
            self.loadTrainExamples(examplesIteration)
            self.skipFirstSelfPlay = skipFirstSelfPlay

            
    def loadTrainExamples(self, iteration):
        """ Loads pre-generated examples from file(s) for the given iteration
            @param iteration: index of iteration from which examples to be loaded 
            
            Examples are loaded from args.numItersForTrainExamplesHistory iterations using the following filename convention 
            filePath is determined as getCheckpointFile(iteration)+".examples.mini"
        """
        modelFile = os.path.join(self.args.load_folder_file[0], getCheckpointFile(iteration))
        
        print("Load trainExamples from MINI file(s)")
        trainExamplesHistory = []
        # check mini files
        for n in range(self.args.numItersForTrainExamplesHistory):
            examplesIteration = iteration-n
            examplesFile = os.path.join(self.args.load_folder_file[0], getCheckpointFile(examplesIteration)+".examples.mini")
            if not os.path.isfile(examplesFile):
                # look for file in checkpoint folder
                examplesFile = os.path.join(self.args.checkpoint, getCheckpointFile(examplesIteration)+".examples.mini")
                if not os.path.isfile(examplesFile):
                    if examplesIteration>=0: 
                        ask_for_continue("MINI file with trainExamples not found:"+examplesFile+". Continue? [y|n]")
                        continue
                    elif examplesIteration<0:
                        break
            print("Load trainExamples from MINI file:", examplesFile)
            with open(examplesFile, "rb") as f:
                iterationExamples = Unpickler(f).load()
                print("...loaded ", len(iterationExamples), "examples")
                trainExamplesHistory.insert(0, iterationExamples) 
            f.closed

        print("length of trainExamplesHistory:", len(trainExamplesHistory))
        for examples in trainExamplesHistory:
            print("Length of mini pack of examples:", len(examples))
        return trainExamplesHistory

    def ensureMaxCheckpoints(self):
        max = self.args.maxCheckpointFiles
        num = 0
        for i in range(self.args.numIters):
            filename = self.args.checkpoint + getCheckpointFile(i)
            if os.path.isfile(filename):
               num += 1
        while num >= max:
            for i in range(self.args.numIters):
                filename = self.args.checkpoint + getCheckpointFile(i)
                if os.path.isfile(filename):
                   os.remove(filename)
                   num -= 1
        
