from __future__ import print_function

import os, sys, time
import numpy as np
from collections import deque

from checkers.CheckersLogic import Move


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

def getCheckpointFile(iteration):
    return 'checkpoint_' + str(iteration) + '.pth.tar'
        
def calc_split(samples_length, batch_size):
    """ Calculates validation split near to 0.2 using length of samples and batch_size.
        Train and validation parts must be a multiple of batch_size.
        For example, samples = 27648, batch = 512
                split = 0.2037(037), train on 22016, validate on 5632
         """
    split = round(samples_length * 0.125 / batch_size) * batch_size / samples_length
    while(True):
        split_at = int(samples_length * (1. - split)) # this is from Keras model.fit()
        if split_at % batch_size == 0:
            break
        split = float(str(split)[:-1])
        #print(split)
    return split

def clean_folder(folder):
    """ Cleans the given folder, i.e. removes all files """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

def user_input(text=""):
    if sys.version_info < (3,0):
        return raw_input(text)
    return input(text)

def ask_for_continue(text):
    """ Prompt user for continue/exit
            press 'y' for continue or 'n' for exit
    """
    # print text to stderr
    print(text, file=sys.stderr)
    # print the same to stdout
    r = user_input(text)
    if r!="y":
        sys.exit()
        
def read_file(filePath):
    with open(filePath, "r") as f:
        examples = f.read()
    f.closed
    
def write_file(filePath, obj):
    with open(filePath, "w+") as f:
        print(obj, file=f)
    f.closed    
        
def debug_pi(pi):
    """ Debugs policy vector - prints all indices where pi>0 as well as all corresponding values """
    nonzero = np.where(pi>0)[0]
    print(nonzero)
    print(pi[nonzero])

def check_stop_condition():
    """ Check the semaphore which could be raised by user. If yes, the program exits immediately. """
    
    if os.path.isfile("stop"):
        print("STOP signal detected. Exit.")
        #self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='final.pth.tar')
        sys.exit()

def shrinkToBeMultiplierOf(list, number):
    """ Shrinks list for its size to be multiple of the given number """
    assert len(list)>=number, "too few items in the list: "+str(len(list)) 
    remainder = len(list) % number
    if remainder>0:
        for _ in range(remainder):
            list.pop()
            
def get_pieces_as_str(pieces):
    """ Returns string representation of the board for MCTS 
        @param classic: whether board should be rotated to the classic orientation (useful for repetition count)
    """
    a = np.reshape(pieces, 8*8)
    # convert black pieces (-1,-3) to (7,5)
    s = ''.join([str(x if x>=0 else x+8) for x in a])
    return s

def strRepr(position):
    """ position is np.array of 8 image planes: 'wW bB P nkr'  """
    
    LONG_CAPTURE = 4
    NO_PROGRESS = 5
    KING_MOVES = 6
    REPETITION = 7
    
    #if np.sum(position[1], (0,1))==0:
    #    continue
    pieces = np.array(position[0], dtype=np.int16, copy=True)
    long_cap = None
    no_progress = 0
    king_moves = 0
    repetition = 0
    for y in range(8):
        for x in range(8):
            pieces[x][y] += 3*position[1][x][y] # white kings
            pieces[x][y] -= position[2][x][y] # black men
            pieces[x][y] -= 3*position[3][x][y] # black kings
            if position[LONG_CAPTURE][x][y] == 1:
                long_cap = (x,y)
            
    no_progress = int(round(position[NO_PROGRESS][0][0] * 128))
    king_moves = int(round(position[KING_MOVES][0][0] * 32))
    repetition = str(int(round(position[REPETITION][0][0] * 4)))
    
    nop = str(hex(no_progress)[2:]).zfill(2)
    kmc = str(hex(king_moves)[2:]).zfill(2)
    rot = "?"
    
    lcs = "--"
    if long_cap:
        lcs = Move(0,0,long_cap[0],long_cap[1],None).dest_str()  
    
    s = repetition + nop + rot + get_pieces_as_str(pieces) + lcs + kmc
    return s

def validate_random_sample(examples):
    # random validation of one position per step
    index = np.random.randint(len(examples))
    s_precalculated = examples[index][3]
    s_calculated = strRepr(examples[index][0])
    assert s_precalculated==s_calculated, "step="+str(step)+"; index="+str(index)+": "+s_precalculated+"=>"+s_calculated 

