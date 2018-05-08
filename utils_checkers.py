from __future__ import print_function

import os, sys, time
import numpy as np
from collections import deque

from checkers.CheckersLogic import Board


def decode_position(s):
    """ s = 100r0010700000000700001000000000000000000000000000071010507000010700--00 """
    repetitions = s[0]
    noprogress = s[1:3]
    rotation = s[3]
    kingmoves = s[-2:]
    longcapture = s[-4:-2]
    pos = s[4:-4]
    a = np.reshape(list(pos), (8,8))
    
    pieces = np.empty([8,8], np.int16)
    
    for y in range(8):
        for x in range(8):
            pieces[x][y] = int(a[x][y])
            if pieces[x][y] > 4:
                pieces[x][y] -= 8 

    b = Board()
    #print("initial position:", b.stringRepr)
    b.pieces = pieces
    if rotation == 'r':
        b.rotation = 180
    b.update_string_repr()
    b.display()
    
    print(repetitions)
    print(noprogress)
    print(rotation)
    print(longcapture)
    print(kingmoves)
    #print(pos)
    #print(a)
