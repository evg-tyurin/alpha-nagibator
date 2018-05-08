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
NeuralNet wrapper class for the CheckersNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

"""

args = dotdict({
    'lr': 0.2,
    'dropout': 0.3,# not used by residual model
    'epochs': 15,
    'batch_size': 32,
    'cuda': False,
    'num_channels': 512,# not used by residual model
    
    'cnn_filter_num': 256, 
    'cnn_first_filter_size': 5,
    'cnn_filter_size': 3,
    'residual_block_num': 5,
    'l2_reg': 1e-4,
    'value_fc_size': 256,
    'trainer_loss_weights': [1.0, 1.0], # not used // [policy, value] prevent value overfit in SL
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.args = args
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples, filePath):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        return history

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = self.game.getImageStack(board)
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.makedirs(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder, filename):
        print("load model from ", folder, filename)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            r = user_input("Model not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
            #raise Exception("No model in path:"+str(filepath))
        else:
            self.nnet.model.load_weights(filepath)

    def prepare_for_mt(self):
        self.nnet.model._make_predict_function()
        #self.graph = tf.get_default_graph()
        
    def recreate(self, folder, filename):
        self.nnet.destroy()
        self.nnet = onnet(self.game, args)
        self.load_checkpoint(folder, filename)
