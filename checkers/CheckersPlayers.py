from __future__ import print_function

import numpy as np
import sys, requests
sys.path.append('..')
from MCTS import MCTS
from .CheckersLogic import Move
from .engine import *
from utils import user_input
"""
Random, NNet and Human-interacting players for the game of Checkers.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

"""
OK = "OK"

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    
    def makeOpponentMove(self, move):
        pass

    def reset(self):
        pass
    
class NNetPlayer():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        #self.mcts = MCTS(game, nnet, args)
        # n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    def play(self, board):
        # print("[",self.mcts.id,"] getActionProb:", self.mcts.numActionProbs+1) #, "took", time, "ms")
        return np.argmax(self.mcts.getActionProb(board, temp=0))
    
    def makeOpponentMove(self, move):
        pass

    def reset(self):
        self.mcts = MCTS(self.game, self.nnet, self.args) # reset search tree        

class HumanCheckersPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        board.display()
        if len(board.executed_moves)>0:
            print("last_move:", board.executed_moves[-1])
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(Move.parse_action(i), end=", ")
        print("")
        while True:
            a = user_input(">>")
            try:
                move = Move.parse(a,board)
            except Exception as e:
                print(e)
                continue
            a = move.move_id
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
    
    def makeOpponentMove(self, move):
        pass

    def reset(self):
        pass

class EngineCheckersPlayer():
    """ 
        Player communicates with 3rd party engine to get next move.
    """
    def __init__(self, game, engineLibName):
        self.game = game
        self.engine = KallistoApiEngine(engineLibName)
        
    def makeOpponentMove(self, moves):
        strMoves = format_to_kallisto_moves(moves)
        self.engine.makeMove(strMoves)

    def play(self, board):
        #board.display()
        strMove = self.engine.think()
        move = Move.parse(strMove,board)
        if board.rotation != 0:
            move = move.rotate()
            move.calc_id()
        
        a = move.move_id

        return a
    
    def reset(self):
        self.engine.reset()

class HttpCheckersPlayer():
    """
        Player communicates with the engine through HTTP.
    """
    
    def __init__(self, game, engineUrl):
        self.game = game # not used
        self.engineUrl = engineUrl
        text = self.send_command("init")
        print(text)
        
    def makeOpponentMove(self, moves):
        """ 
            Sends moves to the engine 
            @param moves: list of moves to be made 
        """
        print("makeMove:", moves)
        strMoves = format_to_kallisto_moves(moves)
        text = self.send_command("make_move?list="+strMoves)
        assert text==OK
        print(text)

    def play(self, board):
        #board.display()
        strMove = self.send_command("think")
        print("engine move:", strMove)

        move = Move.parse(strMove,board)
        if board.rotation != 0:
            move = move.rotate()
            move.calc_id()
        
        a = move.move_id

        return a
    
    def reset(self):
        text = self.send_command("reset")
        assert text==OK
        print(text)
        
    def send_command(self, command):
        print("send: ", command)
        r = requests.get(self.engineUrl+command)
        return r.text

