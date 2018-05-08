from __future__ import print_function

import math
import numpy as np
import os
from shutil import copyfile

class Debug():

    def __init__(self):
        self.folder = os.path.join("Debug","mcts")

    def print_to_file(self, board, filename, header=None):
        """ prints to file board visualization (if not yet) and legal moves"""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        #filename = board.id if board.id else "board.txt"
        file = os.path.join(self.folder, filename)
        header_only = False #os.path.isfile(file)
        with (open(file,'a+')) as f:
            if header:
                print(" === ", header, " === ", file=f)
            else:
                print(" === ??? === ", file=f)
            if header_only:
                print("board.id:", board.id, file=f)
                print("rotation:", board.rotation, ", pieces:", board.count_pieces(), ", halfMoves:", board.halfMoves, ", no-progress:", board.noProgressCount, file=f)
            else:
                board.display(f)
            
            self.print_legal_moves(board, f)
            #print("executed_moves:", board.executed_moves, file=f)
        f.closed 
        return file
    
    def print_legal_moves(self, board, file=None):
        """ prints to file or stdout list of legal_moves of the given board """
        print("legalMoves = [", end="", file=file)
        if board.legal_moves:
            for mv in board.legal_moves:
                if mv: 
                    print(mv, ", ", end="", file=file)
        print("]", file=file)
        
