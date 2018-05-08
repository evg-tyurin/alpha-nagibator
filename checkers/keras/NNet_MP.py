from __future__ import print_function

import numpy as np
import time, datetime, os, sys
sys.path.append('..')
from utils import *
from multiprocessing import connection, Pipe
from threading import Thread
from pytorch_classification.utils import AverageMeter
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import model_from_json
from keras import backend as K

from NeuralNet import NeuralNet
from .CheckersNNet import CheckersNNet as onnet

"""
NeuralNet wrapper class for the CheckersNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 6, 2018.

"""

args = dotdict({
    'lr': 0.02,
    'dropout': 0.3,# not used by residual model
    'epochs': 50,
    'batch_size': 256,
    'cuda': False,
    'num_channels': 512,# not used by residual model
    
    'cnn_filter_num': 256, 
    'cnn_first_filter_size': 5,
    'cnn_filter_size': 3,
    'residual_block_num': 5,
    'l2_reg': 1e-4,
    'value_fc_size': 256,
    'trainer_loss_weights': [1.0, 1.0], # not used // [policy, value] prevent value overfit in SL
    
    'pipe_timeout': 0.004, # 0.01 for self-play, 0.005 or less for Arena 
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.args = args
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.pipes = []
        csvFile = datetime.datetime.now().strftime("NNet.training-%Y-%m-%d_%H-%M-%S.csv")
        self.folder = "Debug"
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        self.csv_filepath = os.path.join(self.folder, csvFile)

    def train(self, examples, filePath):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # reset lr that could be changed by callbacks
        K.set_value(self.nnet.model.optimizer.lr, self.args.lr)
        
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        saveCheckpoint = ModelCheckpoint(filePath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', epsilon=0.02, factor=0.2, patience=2, min_lr=0.0002, verbose=1, mode='min')
        csv_logger = CSVLogger(self.csv_filepath, append=True)
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min')
        callbacks = [saveCheckpoint, reduce_lr, csv_logger, earlyStop]
        split = calc_split(len(target_vs), args.batch_size)
        history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs, validation_split=split, shuffle=True, callbacks=callbacks)
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

    def save_layers(self, folder, filename):
        filepath = os.path.join(folder, filename)
        print("save model layers to", folder, filename)
        if not os.path.exists(folder):
            print("Make directory", folder)
            os.makedirs(folder)
        strJson = self.nnet.model.to_json(indent=4)
        with open(filepath, "w+") as f:
            print(strJson, file=f)
        f.closed

    def load_checkpoint(self, folder, filename):
        print("load model from", folder, filename)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            ask_for_continue("Model not found: "+filepath+". Continue? [y|n]")
            #raise Exception("No model in path:"+str(filepath))
        else:
            self.nnet.model.load_weights(filepath)

    def load_layers(self, folder, filename):
        print("load model layers from ", folder, filename)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            #ask_for_continue("Model not found: "+filepath+". Continue? [y|n]")
            raise Exception("No model in path:"+str(filepath))
        
        with open(filepath, "r") as f:
            strJson = f.read()
        f.closed
        self.nnet.model = model_from_json(strJson)

    def prepare_for_mt(self):
        self.nnet.model._make_predict_function()
        #self.graph = tf.get_default_graph()
        
    def recreate(self, folder, filename):
        self.nnet.destroy()
        self.nnet = onnet(self.game, args)
        self.load_checkpoint(folder, filename)

    def __start(self):
        """
        Starts a thread to listen on the pipe and make predictions
        :return:
        """
        print("NNet_MP.start(), pipe_timeout=", self.args.pipe_timeout)
        self.prepare_for_mt()
        prediction_worker = Thread(target=self._predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipes(self, num):
        """
        Creates a list of pipes on which observations of the game state will be listened for. Whenever
        an observation comes in, returns policy and value network predictions on that pipe.

        :param int num: number of pipes to create
        :return str(Connection): a list of all connections to the pipes that were created
        """
        print("NNet_MP.get_pipes() current pipe count:", len(self.pipes), "; add ", num, "pipes")
        
        self.__start()
        return [self.create_pipe() for _ in range(num)]

    def create_pipe(self):
        """
        Creates a new two-way pipe and returns the connection to one end of it (the other will be used
        by this class)
        :return Connection: the other end of this pipe.
        """
        me, you = Pipe()
        self.pipes.append(me)
        return you

    def _predict_batch_worker(self):
        """
        Thread worker which listens on each pipe in self.pipes for an observation, and then outputs
        the predictions for the policy and value networks when the observations come in. Repeats.
        """
        meter = AverageMeter()
        while True:
            ready = self.wait(self.pipes, timeout=self.args.pipe_timeout)
            if not ready:
                continue
            
            meter.update(len(ready))
            if meter.count % 1000 == 0:
                print("Prediction Worker: count=", meter.count, ", min/avg/max = ", meter.min, meter.avg, meter.max)
            
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    try:
                        (flag,obj) = pipe.recv()
                        if flag==0:
                            print("Stop PIPE")
                            pipe.close()
                            self.pipes.remove(pipe)
                            break
                        data.append(obj)
                        result_pipes.append(pipe)
                    except EOFError:
                        # pipe is closed
                        print("closing pipe...")
                        self.pipes.remove(pipe)
                        break
            if not self.pipes:
                print("There is no PIPE. Prediction worker exits.")
                break
            if not data:
                continue
                #print("There is no DATA. Prediction worker exits.")
                #break
            data = np.asarray(data, dtype=np.float32)
            policy_ary, value_ary = self.nnet.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))
        # print stats
        print("Prediction Worker results: count=", meter.count, ", min/avg/max = ", meter.min, meter.avg, meter.max)

    def wait(self, pipe_list, timeout):
        ready = []
        for pipe in pipe_list:
            if pipe.poll(timeout):
                ready.append(pipe)
        return ready
    