import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .CheckersNNet import CheckersNNet as onnet

"""
Client for contacting NeuralNet in an async way.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 6, 2018.

"""

import numpy as np

class NNet_Client():
    def __init__(self, game, pipe):
        self.game = game
        self.pipe = pipe
        self.id = np.random.randint(1000)

    def predict(self, board):
        """
        Gets a prediction from the policy and value network
        :param board: np array with board
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        
        """
        # timing
        start = time.time()
        
        #print("[",self.id,"] predict(board)")

        # preparing input
        board = self.game.getImageStack(board)
        # board = board[np.newaxis, :, :]

        # 1 is a normal flag, 0 is a stop flag. Stop flag is usually sent from the main thread.
        self.pipe.send((1,board))
        ret = self.pipe.recv()
        return ret

        # run
        #pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        #return pi[0], v[0]

