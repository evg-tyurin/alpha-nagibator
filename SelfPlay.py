from collections import deque
from pytorch_classification.utils import Bar, AverageMeter
import numpy as np
import os, time, datetime

from MCTS import MCTS
from utils import *


"""
Manager for single-threaded Self-Play phase.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 8, 2018.

"""
class SelfPlay():
    """ Manager for single-threaded Self-Play.
            nnet : instance of NNet_MP.NNetWrapper
    """

    def executeEpisodes(self, game, nnet, args, iteration):
        """ Executes a number of episodes specified in args """
        self.game = game
        self.nnet = nnet
        self.args = args

        self.folder = self.args.folder
        
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        eps_time = AverageMeter()
        bar = Bar('Self Play', max=self.args.numEps)
        end = time.time()

        for eps in range(self.args.numEps):
            #print("episode:", eps+1, " of ", self.args.numEps)
            self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
            self.mcts.debug.folder = os.path.join("Debug", str(iteration)+"-"+str(eps))
            iterationTrainExamples += self.executeEpisode(eps)
            # print MCTS stats after we end up with MCTS instance
            self.mcts.print_stats()

            # bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()
        
        return iterationTrainExamples

    def executeEpisode(self, episode):
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
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            #print("step:", episodeStep)
            canonicalBoard = self.game.getCanonicalForm(board,curPlayer)
            canonicalBoard.id = "eps"+str(episode)+"_step"+str(episodeStep)+"_0"
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
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
                board, curPlayer = self.game.getNextState(board, curPlayer, action)
            except AssertionError:
                board.display()
                raise
            
            r = self.game.getGameEnded(board, curPlayer)
            
            # self.check_stop_condition()

            if r!=0:
                self.game.printGameRecord(board, curPlayer, self.folder)
                return [(x[0], x[2], r*((-1)**(x[1]!=curPlayer)), x[3]) for x in trainExamples]
