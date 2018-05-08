from __future__ import print_function

import math
import numpy as np
import os
from shutil import copyfile
import sys
sys.path.insert(0,"timing")
from GS_timing import millis
from pytorch_classification.utils import AverageMeter
from Debug import Debug

import warnings, copy

class MCTS():
    """
    This class handles the MCTS tree.
    """
    
    # (debug) print every board to a file
    print_boards = False

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s - full list of size game.getActionSize
        self.Ls = {}        # stores LegalMoves for board s - short list
        
        self.Nbnodes = {}   # stores number of nodes outgoing from node (board) s
        
        self.numActionProbs = 0
        self.searchIndex = 0
        self.predictionMeter = AverageMeter()
        self.debug = Debug()
        self.maxHalfMovesForDebug = 150
        
        self.id = np.random.randint(1000)

        warnings.filterwarnings('error', category=RuntimeWarning)

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        self.numActionProbs += 1
        
        if not canonicalBoard.id:
            canonicalBoard.id = "ActionProb_"+str(self.numActionProbs)+"_" 
        
        for i in range(self.args.numMCTSSims):
            #print("getActionProb, simulation:", i)
            self.search(canonicalBoard, True)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            maxi = max(counts)
            #print("maxi=",maxi, " of counts=", counts)
            allBest = np.where(np.array(counts)==maxi)[0]
            #print("[",self.id,"] getActionProb:", self.numActionProbs, "allBest=", allBest)
            bestA = np.random.choice(allBest)
            #print("[",self.id,"] getActionProb:", self.numActionProbs, "bestA=", bestA)
            
            # bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        if sum(counts)==0:
            canonicalBoard.display()
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard, isRootNode):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        
        self.searchIndex += 1
        myIndex = self.searchIndex
        #print("MCTS.search: ", myIndex, " size of map: ", len(self.Ps))

        s = self.game.stringRepresentation(canonicalBoard)
        
        # check rotation and active player
        #rot = s[3]
        #assert rot=="r" or rot=="n", "Illegal rotation flag:"+str(rot)+" in "+s
        #if rot=="r":
        #    assert rot=="r" and canonicalBoard.halfMoves % 2 == 1, canonicalBoard.display()
        #else:
        #    assert rot=="n" and canonicalBoard.halfMoves % 2 == 0, canonicalBoard.display()
        

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            if self.Es[s]!=0 and self.print_boards:
                # print("game over detected, r:", self.Es[s], ", halfMoves:", canonicalBoard.halfMoves, ", no-progress:", canonicalBoard.noProgressCount)
                #print("all_moves: ", canonicalBoard.executed_moves)
                #canonicalBoard.display()
                self.debug.print_to_file(canonicalBoard, s, "Game over, result:"+str(self.Es[s]))
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            start = millis()
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            time = millis() - start
            self.predictionMeter.update(time)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            # for debugging
            # orig_Ps_s = copy.deepcopy(self.Ps[s]) 
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            # for debugging
            # masked_Ps_s = copy.deepcopy(self.Ps[s]) 
            #try:
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked
                # make all valid moves equally probable
                print("All valid moves were masked, do workaround. board.id=",canonicalBoard.id)
                #print("valid_moves:", canonicalBoard.filter_legal_moves())
                #print("executed_moves:", canonicalBoard.executed_moves)
                #canonicalBoard.display()
                predicted = "?" # np.where(orig_Ps_s > 0)
                self.debug.print_to_file(canonicalBoard, "workarounds", "All valid moves were masked, do workaround\npredicted_actions:"+str(predicted))

                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            #except Warning:
            #    print("Error detected")
            #    canonicalBoard.display()
            #    self.debug.print_legal_moves(canonicalBoard)
            #    print("all_previous_moves: ", canonicalBoard.executed_moves)
            #    # print to file
            #    self.debug.print_to_file(canonicalBoard, s, "Error detected")
            #    raise
                

            self.Vs[s] = valids
            self.Ls[s] = canonicalBoard.legal_moves
            self.Ns[s] = 0
            
            #if self.print_boards:
            #    self.debug.print_to_file(canonicalBoard, s, "Valid moves calculated")
            
            return -v
        
        canonicalBoard.legal_moves = self.Ls[s]

        valids = self.Vs[s]
        cur_best = -float('inf')
        #best_act = -1
        allBest = []

        # add Dirichlet noise for root node. set epsilon=0 for Arena competitions of trained models
        e = self.args.epsilon
        if isRootNode and e>0:
            noise = np.random.dirichlet([self.args.dirAlpha] * len(canonicalBoard.filter_legal_moves()))
            
        # pick the action with the highest upper confidence bound
        i = -1
        for a in range(self.game.getActionSize()):
            if valids[a]:
                i += 1
                if (s,a) in self.Qsa:
                    q = self.Qsa[(s,a)]
                    n_s_a = self.Nsa[(s,a)]
                    #u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    q = 0
                    n_s_a = 0
                    #u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])     # Q = 0 ?

                p = self.Ps[s][a]
                if isRootNode and e>0:
                    p = (1-e) * p + e * noise[i]
            
                u = q + self.args.cpuct * p * math.sqrt(self.Ns[s]) / (1 + n_s_a)

                if u > cur_best:
                    cur_best = u
                    #best_act = a
                    del allBest[:]
                    allBest.append(a)
                elif u == cur_best:
                    allBest.append(a)

        #a = best_act
        a = np.random.choice(allBest)
        try:
            assert a >= 0, "Illegal action="+str(a)
            next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
            next_s = self.game.getCanonicalForm(next_s, next_player)
            # s1 = self.game.stringRepresentation(next_s)
            next_s.id = self.next_board_id(canonicalBoard, s)
            if next_s.halfMoves > self.maxHalfMovesForDebug:
                self.maxHalfMovesForDebug = next_s.halfMoves
                print("info: board.id:", canonicalBoard.id, "=>", next_s.id, ", pieces:", next_s.count_pieces(), ", halfMoves:", next_s.halfMoves, ", no-progress:", next_s.noProgressCount)
            #if self.print_boards:
            #    self.debug.print_to_file(next_s, s1, "Newly generated position, previous one: "+s)
        except:
            print("Error detected")
            print(s)
            canonicalBoard.display()
            self.debug.print_legal_moves(canonicalBoard)
            # a < 0 occurs sometimes in MCTS.search() 
            move = None if a < 0 else canonicalBoard.parse_action(a)
            print("execute_move:", move)
            print("executed_moves: ", canonicalBoard.executed_moves)
            # print to file
            file = self.debug.print_to_file(canonicalBoard, s, "Error detected")
            with (open(file,'a+')) as f:
                print("execute_move:", move, file=f)
            f.closed
            raise
        
        #if next_player==1:
        #    print("long capture detected")

        # print("search next_state")
        v = self.search(next_s, False)
        # trick for long_captures
        v *= -next_player

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        #print("END OF MCTS.search: ", myIndex, " size of map: ", len(self.Ps))
        return -v

    def next_board_id(self, previousBoard, s_of_previousBoard):
        # define board.id        
        if s_of_previousBoard not in self.Nbnodes:
            self.Nbnodes[s_of_previousBoard] = 0
        else:
            self.Nbnodes[s_of_previousBoard] += 1
        next_id = previousBoard.id + "-" + str(self.Nbnodes[s_of_previousBoard])
        return next_id
    
    def print_stats(self):
        # print collected stats of the instance
        print("") # empty line for the case if the carriage was not returned
        print("MCTS: Ps.size = ", len(self.Ps), ", Es.size = ", len(self.Es), ", Qsa.size = ", len(self.Qsa))
        print("MCTS: actionProbs = ", self.numActionProbs, ", searchCount = ", self.searchIndex)
        pm = self.predictionMeter
        print("MCTS: nnet.predictions: count=", pm.count, ", times(min/avg/max/total) = ", pm.min, pm.avg, pm.max, pm.sum)
        

