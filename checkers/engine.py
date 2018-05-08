from __future__ import print_function

import time
from ctypes import *

from .CheckersLogic import Move

"""
Wrapper for Kallisto engines.
You can use Kallisto engines as a strong opponent for your NN-based players.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Feb 20, 2018.

"""
class KallistoApiEngine():
    """
        Engine that comes with Kallisto API compatible DLL. Runs as a 32 bit application only.
    
        char * KALLISTOAPI EI_GetName();
        char * KALLISTOAPI EI_Think();
        void KALLISTOAPI EI_MakeMove(char *move);
        
        void KALLISTOAPI EI_Initialization(PF_SearchInfo si, int mem_lim);
        void KALLISTOAPI EI_Stop();
        void KALLISTOAPI EI_SetupBoard(char *p);
        void KALLISTOAPI EI_NewGame();
        void KALLISTOAPI EI_SetTimeControl(int time, int inc);
        void KALLISTOAPI EI_SetTime(int time, int otime);
        
        char * KALLISTOAPI EI_PonderHit(char *move);
        void KALLISTOAPI EI_Ponder();
        void KALLISTOAPI EI_OnExit();
        void KALLISTOAPI EI_Analyse();
        void KALLISTOAPI EI_EGDB(EdAccess *eda);
    """

    
    def __init__(self, libName):
        self.mydll = windll.LoadLibrary(libName)
        # mydll = windll.LoadLibrary("Kallisto_4.dll")
        self.define_functions()
        
        name = self.EI_GetName()
        print (name)
        
        # Initialization params are 1) callback function and 2) hashtable size in Megabytes
        self.EI_Initialization(self._pf_search_info, c_int(1)) # _pf_search_info
        print("Engine ", libName, "initialized")
        
        self.nextCapture = []
        self.thinkTime = 0
        
    def reset(self):
        """ Resets state of the engine and prepares it to a new game """
        # this is called later from self.EI_NewGame()
        # self.EI_SetupBoard("bbbbbbbbbbbb........wwwwwwwwwwwww")
        self.EI_NewGame()
        self.EI_SetTimeControl(c_int(0), c_int(1))
        # 10 sec 
        self.EI_SetTime(c_int(1*1000), c_int(10*1000))
        self.nextCapture = []
        self.thinkTime = 0
        
    def think(self):
        """ Thinks and outputs the next move in the Nagibator notation, i.e. one atomic move + pending long captures """
        print("") # new line after engine's printing
        
        # 1 sec - 10 sec
        self.EI_SetTime(c_int(1*1000), c_int(10*1000))
        
        start = time.time()
        if self.nextCapture:
            move = self.nextCapture.pop(0)
            print("engine move:", move)
            return move
        
        move = self.EI_Think()
        print("original engine's move:", move)

        elapsed = time.time() - start
        self.thinkTime += elapsed
        print("total thinkTime:", int(self.thinkTime), "sec")
                
        move = self.format_to_nagibator_move(move)

        print("engine move:", move)
        return move
    
    def get_intermediate_square(self, squareFrom, capture1, capture2):
        """ Calculates intermediate square between two captured pieces """
        x0,y0 = Move.parse_square(squareFrom)
        x1,y1 = Move.parse_square(capture1)
        tmp = Move(x0,y0,x1,y1,None)
        valid_coords = range(0,8) 
        x2,y2 = Move.parse_square(capture2)
        for n in range(1,7):
            x = x1 + n*tmp.direction()[0]
            y = y1 + n*tmp.direction()[1]
            if x in valid_coords and y in valid_coords:
                if abs(x-x2) == abs(y-y2): # this is a diagonal
                    sq = Move.square(x,y)
                    return sq
        return None

        
    def makeMove(self, move):
        """
            Tells engine to make the move of its opponent to sync board position.
            @param move: string representation of the move using the notation of the engine.
        """
        print("makeMove:", move)
        self.EI_MakeMove(move)
    
    def pf_search_info(self, score, depth, speed, pv, cm):
        """     
            Callback function for presenting calculation info from the engine.
            pv - best variant
            cm - currently analyzed move """
        print(".", end="")
        
    def define_functions(self):
        """ Defines DLL functions and their in/out data types which can be used by the script """
         
        PF_SearchInfo = WINFUNCTYPE(None, c_int, c_int, c_int, c_char_p, c_char_p)
        
        # reference the callback to keep it alive
        self._pf_search_info = PF_SearchInfo(self.pf_search_info)
        
        # function definitions
        self.EI_GetName = self.mydll.EI_GetName
        self.EI_GetName.restype = c_char_p
        
        self.EI_Think = self.mydll.EI_Think
        self.EI_Think.argtypes = []
        self.EI_Think.restype = c_char_p
        
        self.EI_MakeMove = self.mydll.EI_MakeMove
        self.EI_MakeMove.argtypes = [c_char_p]
        
        self.EI_Initialization = self.mydll.EI_Initialization
        self.EI_Initialization.argtypes = [PF_SearchInfo, c_int]
        
        self.EI_SetupBoard = self.mydll.EI_SetupBoard
        self.EI_SetupBoard.argtypes = [c_char_p]
        
        self.EI_NewGame = self.mydll.EI_NewGame
        self.EI_NewGame.argtypes = []
        
        self.EI_Stop = self.mydll.EI_Stop
        self.EI_Stop.argtypes = []
        
        self.EI_SetTimeControl = self.mydll.EI_SetTimeControl
        self.EI_SetTimeControl.argtypes = [c_int, c_int]
        
        self.EI_SetTime = self.mydll.EI_SetTime
        self.EI_SetTime.argtypes = [c_int, c_int]
        
        #self.EI_Ponder = self.mydll.EI_Ponder();
        #self.EI_OnExit = self.mydll.EI_OnExit();
        #self.EI_Analyse = self.mydll.EI_Analyse();
        #self.EI_EGDB = self.mydll.EI_EGDB(EdAccess *eda);

    def format_to_nagibator_move(self, move):
        """ Reformats the move from Kallisto API format to Nagibator format """
        print("reformat:", move)
        if len(move)==4:
            move = move[:2]+"-"+move[-2:]
        elif ':' in move: 
            """ KallistoAPI engines use long format for captures like g5:f4:e3 where 'f4' is captured piece """
            squares = move.split(":")
            assert len(squares)>=3, "KallistoAPI engine should use long format for captures: "+move
            if len(squares)>3:
                """ for example, capture of three pieces at b4, d2 and g3 - a5:b4:d2:g3:h4 """
                captures = squares[1:-1]
                new_list = list(captures)
                # insert intermediate squares between captures
                # get square after sq[1] from which sq[2] can be captured too
                start = squares[0]
                for n in range(1,len(captures)):
                    sq = self.get_intermediate_square(start, captures[n-1], captures[n])
                    assert sq, "Can't parse engine move:"+move
                    new_list.insert(2*n-1, sq)
                    start = sq
                new_list.insert(0, squares[0])
                new_list.append(squares[-1])
                move = new_list[0]+":"+new_list[2]
                for n in range(1,len(new_list)/2):
                    next = new_list[2*n]+":"+new_list[2*n+2]
                    self.nextCapture.append(next)
                return move
            else: # len=3 
                move = squares[0]+":"+squares[2]
        return move

def format_to_kallisto_moves(moves):
    """ 
        Formats moves to KallistoAPI form
        @param moves: list of Move objects  
    """
    if len(moves)==1:
        m = moves[0]
        strMove = str(m)
        if '-' in strMove:
            strMove = strMove.replace("-","")
        elif ':' in strMove:
            strMove = strMove[:2]+":"+Move.square(m.capture[0],m.capture[1])+":"+strMove[-2:]
    else:
        strMove = Move.square(moves[0].x0,moves[0].y0)
        for m in moves:
            strMove += ":"+Move.square(m.capture[0],m.capture[1])
        strMove += ":"+Move.square(moves[-1].x1,moves[-1].y1)
    return strMove

