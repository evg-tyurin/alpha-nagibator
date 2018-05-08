from __future__ import print_function

import numpy as np

'''
Board class for the game of Checkers (Russian variant).

Board size is 8x8.
Board data:
  1=white man, -1=black man, 
  3=white king, -3=black king,
  0=empty
  first dim is column , 2nd is row:
     pieces[0][0] is the top left square,
     pieces[7][0] is the bottom left square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

'''
# from bkcharts.attributes import color
class Board():

    # list of all 4 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,-1),(-1,-1),(-1,1)]
    
    xx = ['a','b','c','d','e','f','g','h']


    def __init__(self, board=None):
        "Set up initial board configuration."

        # few methods of Board and Move support size=8 only
        self.n = 8
        self.id = ""
        # half-moves with no-progress
        self.noProgressCount = 0
        # number of king-pieces
        self.whiteKingCount = 0
        self.blackKingCount = 0 
        # number of half-moves made by kings only
        self.kingMoveCount = 0
        
        self.last_long_capture = None
        
        self.classic_pieces_str = None
        
        if board:
            self.pieces = np.array(board.pieces, dtype=np.int16, copy=True)
            self.rotation = board.rotation
            self.halfMoves = board.halfMoves
            self.executed_moves = list(board.executed_moves)
            
            # repetitions of the same position
            self.positionCount = board.positionCount.copy()
            self.noProgressCount = board.noProgressCount
            self.whiteKingCount = board.whiteKingCount            
            self.blackKingCount = board.blackKingCount 
            self.kingMoveCount = board.kingMoveCount
            
            self.last_long_capture = board.last_long_capture
            self.stringRepr = board.stringRepr
            self.classic_pieces_str = board.classic_pieces_str
        else:
            # Create the empty board array.
            self.pieces = np.empty([8,8], np.int16)
    
            # Set up the initial pieces.
            for y in range(self.n):
                for x in range(self.n):
                    if (y==0 or y==2) and x % 2==0:
                        self.pieces[x][y] = 1
                    elif y==1 and x % 2 == 1:
                        self.pieces[x][y] = 1
                    elif (y==5 or y==7) and x % 2==1:
                        self.pieces[x][y] = -1
                    elif y==6 and x % 2 == 0:
                        self.pieces[x][y] = -1
                    else:
                        self.pieces[x][y] = 0
            # view: a player to which the board is oriented
            #     1: white, -1: black, values are the same as player colors        
            self.rotation = 0
            self.halfMoves = 0
            self.executed_moves = []
            
            # repetitions of the same position
            self.positionCount = {}
            self.update_position_counter()
            self.stringRepr = self.__stringRepresentation()
                    
        self.legal_moves = None

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used,  Arena always invokes for color=1        
        """
        
        #print("get_legal_moves: color: ", color)
        #self.display()
        
        moves = set()  # stores the legal moves.
        if self.last_long_capture:
            # player must continue his long capture 
            start = self.last_long_capture
            #if self.rotation != 0 or color == -1:
            #    start = start.clone().rotate()
            newmoves = self.get_atomic_moves_for_square(start.x1, start.y1, True)
            moves.update(newmoves)
        else:
            captures_only = False
            # Get all the empty squares (color==0)
            for y in range(self.n):
                for x in range(self.n):
                    if self[x][y] * color > 0:
                        try:
                            newmoves = self.get_atomic_moves_for_square(x, y, captures_only)
                            # if there are any captures then all simple moves are ignored
                            if not captures_only:
                                if len(newmoves)>0 and newmoves[0].capture:
                                    moves.clear() 
                                    captures_only = True                        
                            moves.update(newmoves)
                        except AssertionError:
                            print("AssertionError detected")
                            self.display()
                            raise
                    
        self.legal_moves = self.to_array(moves)
        return list(moves)
    
    def to_array(self, moves):
        arr = [None]*(Board.get_action_size())
        for move in moves:
            assert move.move_id < Board.get_action_size(), self.display()+str(move)+",id="+str(move.move_id)
            assert move.move_id >=0, self.display()+str(move)+",id="+str(move.move_id)
            arr[move.move_id] = move
        return arr
    
    def get_classic_piece(self, x, y):
        if self.rotation==0:
            return self[x][y]
        return self[7-x][7-y]
    
    @staticmethod
    def get_action_size():
        # the method supports n=8 only
        # assert self.n == 8
        # action space
        # square_from = self.n*self.n/2 = 8x8/2 = 32 (for n=8, 5 bits)
        # move vector, max of 7 squares in 4 directions = (self.n-1)*4 = 7*4 = 28 (for n=8)
        # maxNumberOfMoves = 32*28 = 896
        # to simplify action encoding/decoding take max as bin(28)_bin(32) = 11100_11111 = 927
        return 927
    
    def get_atomic_moves_for_square(self, x, y, captures_only=False):
        moves = self.get_moves_for_square(x, y, captures_only)
        #for m in moves:
        #    m.is_long_capture = (m.additional_capture != None)
        #    m.additional_capture = None
        return moves
        
    def get_moves_for_square(self, x, y, captures_only=False):
        """ Returns all legal moves for the given square
        """
        piece = self[x][y]
        assert piece != 0, str(x)+","+str(y)+self.display()
        moves = set()
        capture_found = False
        promotion_rank = 7 if piece == 1 else 0 # if self.view == 1 else 0
        top_direction = 1 if piece == 1 else -1 # self.view
        # moves of men
        if piece==1 or piece==-1: # man
            for direction in self.__directions:
                x1, y1 = x + direction[0], y + direction[1]
                if not self.is_on_board(x1,y1):
                    continue
                if self[x1][y1]==0:
                    # men move forward only 
                    if direction[1] == top_direction and not captures_only:
                        # promotion = y1==promotion_rank 
                        moves.add(Move(x,y, x1,y1, None))
                elif self[x1][y1] * piece < 0: # possible capture
                    """ TODO man must capture all the pieces on his path """
                    cap_x, cap_y = x1 + direction[0], y1 + direction[1]
                    if self.is_on_board(cap_x, cap_y) and self[cap_x][cap_y]==0:
                        """ capture found! """
                        if not capture_found:
                            moves.clear()
                            capture_found = True         
                            captures_only = True
                        # promotion = cap_y==promotion_rank
                        newmove = Move(x,y, cap_x,cap_y, (x1,y1))
                        #backward = (-direction[0], -direction[1])
                        #captured = [(x1,y1)]
                        #if promotion:
                        #    self.add_king_captures_for_square(cap_x,cap_y,backward,captured,newmove,piece)
                        #else:
                        #    self.add_man_captures_for_square(cap_x,cap_y,backward,captured,newmove,piece)
                        moves.add(newmove)
        # moves of kings
        if piece==3 or piece==-3: # king
            for direction in self.__directions:
                for length in range(1,8):
                    break_length = False
                    x1, y1 = x + length*direction[0], y + length*direction[1]
                    if not self.is_on_board(x1,y1):
                        break # break LENGTH
                    if self[x1][y1]==0:
                        if not captures_only:                             
                            moves.add(Move(x,y,x1,y1, None))
                    elif self[x1][y1] * piece > 0: # same color piece, break searching direction
                        break # break LENGTH                        
                    elif self[x1][y1] * piece < 0: # possible capture
                        """ TODO king must capture all the pieces on his path """
                        captures_in_this_direction = []
                        maxNumCaptures = 0
                        for length2 in range(length+1,8):
                            cap_x, cap_y = x + length2*direction[0], y + length2*direction[1]
                            if self.is_on_board(cap_x, cap_y) and self[cap_x][cap_y]==0:
                                """ capture found! """
                                if not capture_found:
                                    moves.clear()
                                    capture_found = True         
                                    captures_only = True
                                newmove = Move(x,y, cap_x,cap_y, (x1,y1))
                                backward = (-direction[0], -direction[1])
                                captured = [(x1,y1)]
                                is_long_capture = self.has_additional_king_captures_for_square(cap_x,cap_y,backward,captured,piece)
                                """ filter king's moves, the longest capture takes precedence 
                                    among the moves starting from the same square in the same direction """
                                numCaptures = int(is_long_capture)
                                if numCaptures > maxNumCaptures:
                                    for mv in captures_in_this_direction:
                                        moves.remove(mv)
                                    del captures_in_this_direction[:]
                                    maxNumCaptures = 0
                                elif numCaptures < maxNumCaptures: # don't add this move
                                    continue
                                moves.add(newmove)
                                captures_in_this_direction.append(newmove)
                                if numCaptures > maxNumCaptures:
                                    maxNumCaptures = numCaptures
                            else:
                                break_length = True
                                break
                    if break_length:
                        break
            
        return list(moves)
    
    def has_additional_king_captures_for_square(self, x, y, exceptDirection, exceptPieces, color):
        """ Checks whether there is additional capture for King from the given square (x,y).
            x,y: starting square
            exceptDirection: exclude given direction from the search 
            color: color of the player which does move
        """
        for direction in self.__directions:
            if direction == exceptDirection:
                continue
            for length in range(1,8):
                break_length = False
                x1, y1 = x + length*direction[0], y + length*direction[1]
                if not self.is_on_board(x1,y1):
                    break # break LENGTH
                if self[x1][y1]==0: # empty square
                    continue
                if self[x1][y1] * color > 0: # same color piece, break searching direction
                    break # break LENGTH                        
                if self[x1][y1] * color < 0: # possible capture
                    for length2 in range(length+1,8):
                        cap_x, cap_y = x + length2*direction[0], y + length2*direction[1]
                        if self.is_on_board(cap_x, cap_y) and self[cap_x][cap_y]==0 and not (x1,y1) in exceptPieces:
                            """ capture found! """
                            # add_move = Move(x,y, cap_x,cap_y)
                            return True
                            #backward = (-direction[0], -direction[1])
                            #captured = [(x1,y1)]
                            #captured.extend(exceptPieces)
                            #self.add_king_captures_for_square(cap_x,cap_y,backward,captured,add_move,color)
                            
                        else:
                            break_length = True
                            break
                if break_length:
                    break
        return False
        
    def has_additional_man_captures_for_square(self, x, y, exceptDirection, exceptPieces, color):
        """ Checks whether there is additional capture for Man from the given square (x,y).
            x,y: starting square
            exceptDirection: exclude given direction from the search 
            color: color of the player which does move
        """
        piece = color
        promotion_rank = 7 if piece == 1 else 0 # if self.view == 1 else 0
        for direction in self.__directions:
            if direction == exceptDirection:
                continue
            x1, y1 = x + direction[0], y + direction[1]
            if not self.is_on_board(x1,y1):
                continue
            if self[x1][y1]==0: # empty square
                continue
            if self[x1][y1] * color > 0: # same color piece, break searching direction
                continue                         
            if self[x1][y1] * color < 0: # possible capture
                cap_x, cap_y = x1 + direction[0], y1 + direction[1]
                if self.is_on_board(cap_x, cap_y) and self[cap_x][cap_y]==0 and not (x1,y1) in exceptPieces:
                    """ capture found! """                            
                    #promotion = cap_y==promotion_rank
                    # add_move = Move(x,y, cap_x,cap_y, (x1,y1), promotion, rotation=self.rotation)
                    return True
                    #backward = (-direction[0], -direction[1])
                    #captured = [(x1,y1)]
                    #captured.extend(exceptPieces)
                    #if promotion:
                    #    self.add_king_captures_for_square(cap_x,cap_y,backward,captured,add_move,color)
                    #else:
                    #    self.add_man_captures_for_square(cap_x,cap_y,backward,captured,add_move,color)
                    #move.additional_capture = add_move
                    #move.calc_id()
                else:
                    continue
        return False
        
    
    def is_on_board(self, x, y):
        return x>=0 and x<self.n and y>=0 and y<self.n

    def count_pieces(self):
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]!=0:
                    count += 1
        return count
    
    def get_game_result(self, color):
        """ Check whether the game is over (previous method name was is_win() ).
            Check whether the given player has captured all opponent's pieces
            or given player's pieces have valid moves
        @param color (1=white,-1=black) - player to move
        @return: 1: if the given player won, -1: if the given player lost, 0: if game continues
        """
        #print("is_win(", color, ")")
        #self.display()
        
        enemy = -color
        enemyCount = 0
        myValidMoves = 0
        
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] * enemy > 0: 
                    enemyCount += 1
                elif self[x][y] * color > 0:
                    myValidMoves += len(self.get_moves_for_square(x, y, False))
                
                if enemyCount > 0 and myValidMoves > 0:
                    return 0 # game continues
        # FIXME *= color
        if enemyCount==0:
            return 1 # win!
        if myValidMoves==0:
            return -1 # lost
        # check that current player has a valid move
        
        # print("is_win(", color, ") = ", win)
        return 0 # game continues
    
    def validate_move(self, move, color):
        if not move.capture:
            assert move.length() == 1 or abs(self[move.x0][move.y0])==3, "Illegal move: "+str(move)

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        
        move.check_capture(self)
        
        self.validate_move(move, color)
        
        if self.last_long_capture:
            last = self.last_long_capture
            try:
                assert last.x1 == move.x0 and last.y1 == move.y0, "Long capture must continue: "+self.display()+str(last)+" != "+str(move)
            except:
                print("Hello") 

        # validate given move
        #ok = False
        #for lm in self.legal_moves:
        #    if lm and str(lm) == str(move):
        #        ok = True
        #assert ok, "Illegal move: "+str(move)+"; legal_moves: "+str(self.filter_legal_moves())
                
        #num = str(1+int(self.halfMoves/2))+"."
        #if self.halfMoves % 2 != 0:
        #    num += " .. "
        
        addedHalfMoves = self.execute_move_on_pieces(self.pieces, move, color)
        # 1 - for simple moves, 0 - for incompleted long captures
        self.halfMoves += addedHalfMoves
        
        classicMove = move if self.rotation == 0 else move.clone().rotate()        
        self.executed_moves.append(classicMove)
        
        # update position counters
        self.update_position_counter()
        self.stringRepr = self.__stringRepresentation()
        
        #self.display()
        return addedHalfMoves
        
    def execute_move_on_pieces(self, pieces, move, color):
        # check that current player does move
        # print("assert ", pieces[move.x0][move.y0], color)
        # assert pieces[move.x0][move.y0] * color > 0, "player="+str(color)
        assert pieces[move.x0][move.y0] != 0, self.get_debug_info(move)+"\ncolor="+str(color)+"\npiece="+str(pieces[move.x0][move.y0])
        assert pieces[move.x0][move.y0] * color > 0, self.get_debug_info(move)+"\ncolor="+str(color)+"\npiece="+str(pieces[move.x0][move.y0])
        
        if abs(pieces[move.x0][move.y0])==3:
            self.kingMoveCount += 1
        else:
            self.kingMoveCount = 0
        
        promotion = (move.y1 == 7 and pieces[move.x0][move.y0]==1) or (move.y1 == 0 and pieces[move.x0][move.y0]==-1) 
        if move.capture or promotion:
            self.noProgressCount = 0
            self.kingMoveCount = 0
        else:
            self.noProgressCount += 1
        
        pieces[move.x1][move.y1] = pieces[move.x0][move.y0]
        pieces[move.x0][move.y0] = 0
        if promotion:
            pieces[move.x1][move.y1] *= 3
            if pieces[move.x1][move.y1] > 0:
                self.whiteKingCount += 1
            else:
                self.blackKingCount += 1
            # reset position counters   
            self.positionCount.clear()
        
        addedHalfMoves = 1
        
        if move.capture:
            x,y = move.capture
            assert pieces[x][y] != 0, "Start square is empty. "+self.get_debug_info(move)
            if pieces[x][y] == 3:
                self.whiteKingCount -= 1
            elif pieces[x][y] == -3:
                self.blackKingCount -= 1
            pieces[x][y] = 0         
            # reset position counters   
            self.positionCount.clear()
            exceptDirection = move.backward_direction()
            exceptPieces = [(x,y)]
            if abs(pieces[move.x1][move.y1]) == 3:
                is_long_capture = self.has_additional_king_captures_for_square(move.x1, move.y1, 
                                    exceptDirection, exceptPieces, color)
            else:
                is_long_capture = self.has_additional_man_captures_for_square(move.x1, move.y1, 
                                    exceptDirection, exceptPieces, color)
                
            if is_long_capture:
                self.last_long_capture = move
                addedHalfMoves = 0
            else:
                self.last_long_capture = None
        #if move.additional_capture:
        #    self.execute_move_on_pieces(pieces, move.additional_capture, color)
        return addedHalfMoves
    
    def transform_action_for_board(self, action, board):
        """ Transforms the given action received from the current instance of Board for using by the given board """
        if self.rotation != board.rotation:
            m = Move.parse_action(action)
            m = m.rotate()
            return m.calc_id()
        else:
            return action             
            
    def get_debug_info(self, move):
        return "board.id="+str(self.id)+"\n"+self.display()+"legalMoves: "+str(self.filter_legal_moves())+"\nexecute_move: "+str(move)
        
    def __get_move_from_action(self, action):
        #assert self.legal_moves
        #assert self.legal_moves[action], str(self.filter_legal_moves())+", "+str(action)+", "+str(Move.parse_action(action))
        #return self.legal_moves[action]
        return Move.parse_action(action, self)
    
    def get_id(self):
        id = self.__str__()
        id = id[id.find('at')+3:id.find('>')]
        return id
    
    def get_pieces_as_str(self, classic=False):
        """ Returns string representation of the board for MCTS 
            @param classic: whether board should be rotated to the classic orientation (useful for repetition count)
        """
        if classic and self.rotation!=0:
            classic_pieces = np.rot90(self.pieces, 2)
            a = np.reshape(classic_pieces, self.n*self.n)
            # invert colors. convert black pieces (-1,-3) to (7,5)
            s = ''.join([str(-x if x<=0 else -x+8) for x in a])
        else:
            a = np.reshape(self.pieces, self.n*self.n)
            # convert black pieces (-1,-3) to (7,5)
            s = ''.join([str(x if x>=0 else x+8) for x in a])
        return s
    
    def display(self, file=None):
        
        board = self
        # mirror multiplier for pieces, either 1 or -1
        m = 1 if board.rotation == 0 else -1 
        n = board.n

        print("board.id:", self.id, "|", file=file)
        print("stringRepr:", self.stringRepr, file=file)
        print("rotation:", self.rotation, ", pieces:", self.count_pieces(), ", kings: (", self.whiteKingCount, ",", self.blackKingCount, ")",  
              ", halfMoves:", self.halfMoves, ", no-progress:", self.noProgressCount, 
              ", king-moves:", self.kingMoveCount, ", repetitions:", self.get_repetition_count(), file=file)
        print("last_long_capture:", self.last_long_capture, file=file)
        print("  ------------------", file=file)
        for y in range(n-1,-1,-1):
            print(y+1, "|",end="", file=file)    # print the row #
            for x in range(n):
                piece = board[x][y]    # get the piece to print
                if piece*m == -1: print("x ",end="", file=file)
                elif piece*m == 1: print("o ",end="", file=file)
                elif piece*m == -3: print("X ",end="", file=file)
                elif piece*m == 3: print("O ",end="", file=file)
                else:
                    if x==n:
                        print("-",end="", file=file)
                    else:
                        print("- ",end="", file=file)
            print("|", file=file)
    
        print("  ------------------", file=file)
        print("   ",end="", file=file)
        for x in range(n):
            print (self.xx[x]+" ",end="", file=file)
        print("", file=file)
        print("executed_moves:", self.executed_moves, file=file)
        
        return ""
    
    def filter_legal_moves(self):
        """ Returns list of valid moves, that is filtered list of legal_moves excluding None items """
        list = []
        if self.legal_moves:
            for mv in self.legal_moves:
                if mv: 
                    list.append(mv)
        return list
    
    def get_repetition_count(self):
        s = self.classic_pieces_str
        try:
            return self.positionCount[s]
        except KeyError:
            print("Existing keys:")
            for s in self.positionCount.keys():
                print(s)
            print("executed_moves: ", self.executed_moves)
            raise
    
    def update_position_counter(self):
        s = self.get_pieces_as_str(classic=True)
        self.classic_pieces_str = s
        if s in self.positionCount:
            self.positionCount[s] += 1
        else:
            self.positionCount[s] = 1
    
    def __stringRepresentation(self):
        """ String representation of the board for MCTS map keys """
        # board.pieces is 8x8 numpy array (canonical board)
        #hmv = hex(board.halfMoves)[2:].zfill(2) # 31 => 0x1F => 1F
        noProgressCount = self.noProgressCount if self.count_pieces() <= 7 else 0
        nop = str(hex(noProgressCount)[2:]).zfill(2)
        # rotation flag
        rot = "r" if self.rotation != 0 else "n"
        # long capture square - where l.c. continues from
        lcs = self.last_long_capture.dest_str() if self.last_long_capture else "--"
        # king's move count
        kmc = str(hex(self.kingMoveCount)[2:]).zfill(2)
        # repetition count
        repCount = self.get_repetition_count() 
        assert repCount < 4, "repetitionCount="+str(repCount)+self.display()+str(self.executed_moves)
        rep = str(repCount)
        """ active side (player) may be included in the string 
                though... white always move """
        return rep+nop+rot+self.get_pieces_as_str()+lcs+kmc    
    
    def update_string_repr(self):
        """ Updates stringRepr of the board for example after board rotation """ 
        self.stringRepr = self.__stringRepresentation()
        
    def parse_action(self, action):
        return Move.parse_action(action)
    

class Move():
    __directions = [(1,1),(1,-1),(-1,-1),(-1,1)]
    """
        The Move class supports board.size == 8 only
        x0,y0 = start square
        x1,y1 = destination square
    """
    def __init__(self, x0, y0, x1, y1, capture):
        (self.x0, self.y0) = (x0,y0)
        (self.x1, self.y1) = (x1,y1)
        self.capture = capture
        # recalc id if additional_capture are added 
        self.calc_id()
        
    def calc_id(self):
        """ Calculates move_id. 
            MUST be invoked after full initialization including additional_capture """ 
        # move_id is a bitmask of length 10
        # 11100  11111
        # vector from 
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        direction = 0
        if dx>0 and dy>0:
            direction = 0
        elif dx>0 and dy<0:
            direction = 1
        elif dx<0 and dy<0:
            direction = 2
        else:
            direction = 3
        start = int((8 * self.y0 + self.x0)/2) # 1d-index of black squares
        length = abs(dx) 
        # our vector is an index of actual vector in (direction,length)-space
        # actual vector => vector index 
        # (direction=0,length=1) => 0 
        # (direction=1,length=1) => 1 
        # (direction=2,length=1) => 2 
        # (direction=3,length=1) => 3 
        # (direction=0,length=2) => 4, etc. 
        vector = direction + (length-1)*4
        # long_capture = int(self.additional_capture != None or self.is_long_capture)        
        self.move_id = (vector << 5) | start 
        return self.move_id
     
    """   
    def num_captures(self):
        # Returns number of captures in the move
        num = 1 if self.capture else 0
        if (self.additional_capture):
            num += self.additional_capture.num_captures()
        return num"""
    
    def clone(self):
        """ deep copy of the move """
        move = Move(self.x0,self.y0,self.x1,self.y1,None)
        #if self.additional_capture:
        #    move.additional_capture = self.additional_capture.clone()
        #move.is_long_capture = self.is_long_capture
        move.calc_id()
        return move
    
    def backward_direction(self):
        return (Move.sign(self.x0 - self.x1), Move.sign(self.y0 - self.y1))
    
    def direction(self):
        return (Move.sign(self.x1 - self.x0), Move.sign(self.y1 - self.y0))
    
    def length(self):
        return abs(self.x1 - self.x0)
    
    def check_capture(self, board):
        """ Check whether the move is a capture. If yes, set self.capture to appropriate value """
        piece = board[self.x0][self.y0]
        length = abs(self.x1-self.x0)
        if length<2:
            return
        dir = self.direction()
        for i in range(1,length):
            capture = (self.x0 + i*dir[0], self.y0 + i*dir[1])
            captured_piece = board[capture[0]][capture[1]]
            assert Move.sign(piece) != Move.sign(captured_piece), "Illegal jump over own piece:"+str(self) 
            if piece * captured_piece < 0:
                self.capture = capture
                break
    
    def rotate(self):
        """ Rotates move coordinates 180 degrees, as we detect the board was rotated """
        self.x0 = 7 - self.x0
        self.y0 = 7 - self.y0
        self.x1 = 7 - self.x1
        self.y1 = 7 - self.y1
        #if self.capture:
        #    x,y = self.capture
        #    x = 7 - x
        #    y = 7 - y
        #    self.capture = (x,y)
        #if self.additional_capture:
        #    self.additional_capture.rotate()
        #self.rotation = (self.rotation + 180) % 360
        return self
            
    def strdst(self):
        """ string representation of the destination point 
            probably with promotion sign and additional captures """
        dst = self.dest_str()
        # promo = "!" if self.promotion else ""
        #add = "" if not self.additional_capture else ":"+self.additional_capture.strdst()
        #add2 = "(+)" if self.is_long_capture else ""
        return dst
        
    def tostring(self):
        #if self.rotation == 180:
        #    self.rotate()
        src = Board.xx[self.x0]+str(self.y0+1)
        delim = ":" if self.capture else "-"
        dst = self.strdst()
        # rot = "(r)" if self.rotation != 0 else ""
        return src + delim + dst

    def __str__(self):
        return self.tostring()
    
    def __repr__(self):
        return self.tostring()
    
    def dest_str(self):
        """ Short string representation of the destination point, like c3 or h8. 
            Used in stringRepresentation """
        return Board.xx[self.x1]+str(self.y1+1)
    
    @staticmethod
    def square(x, y):
        return Board.xx[x]+str(y+1)
    
    @staticmethod
    def parse(s, board):
        """ Parses a move from the given string
            s = a1-b2 or h6:f8! or a1:c3:e1 etc. """
        #rotation = 0
        long_capture = False
        #if s.endswith("(r)"):
        #    rotation = 180
        #    s = s[:-3]
        if s.endswith("(+)"):
            long_capture = True
            s = s[:-3]
        if '-' in s:
            d = s.split("-")
            x0, y0 = Move.parse_square(d[0])
            x1, y1 = Move.parse_square(d[1])
            piece = board.get_classic_piece(x0, y0) # piece to move
            assert piece, "Start square is empty: "+s
            assert abs(x1-x0) == abs(y1-y0), "Not a diagonal: "+s 
            #assert abs(x1-x0)==1 and abs(y1-y0)==1, "It's possibly king's move: "+s
            length = abs(x1-x0)
            #if length>1:
            #    # this may be a capture
            #    direction = (Move.sign(x1-x0), Move.sign(y1-y0))
            #    for i in range(1,length):
            #        capture = (x0+i*direction[0], y0+i*direction[1])
            #        board.get_classic_piece(capture[0],capture[1])
            #        assert board.get_classic_piece(capture[0],capture[1])==0, "Illegal notation for capture, use colon (:) sign"
            #    # long move which is not a capture can be done by King only
            #    assert abs(piece)==3, "Long moves allowed for Kings only"
            #promotion = s.endswith('!')
            m = Move(x0, y0, x1, y1, None)
        elif ':' in s:
            d = s.split(":")
            m = Move.parse_jump(d[0], d[1], board)
            piece = board.get_classic_piece(m.x0, m.y0) # piece to move
            assert piece, "Start square is empty: "+s
            # additional captures
            if len(d)>2:
                m.additional_capture = Move.parse_jump(d[1], d[2], board)
                if len(d)>3:
                    m.additional_capture.additional_capture = Move.parse_jump(d[2], d[3], board)
        else:
            raise Exception(s)
        m.is_long_capture = long_capture
        #if rotation==180:
        #    m.rotate()
        #    m.rotation = 0
        m.calc_id()
        return m
    
    @staticmethod
    def parse_jump(s1, s2, board):
        """ Parses a single jump from s1 to s2 with capture """
        x0, y0 = Move.parse_square(s1)
        x1, y1 = Move.parse_square(s2)
        promotion = s2.endswith('!')
        piece = board.get_classic_piece(x0, y0) # piece to move
        assert abs(x1-x0) == abs(y1-y0), "Not a diagonal: "+s1+":"+s2
        # assert abs(x1-x0)==2 and abs(y1-y0)==2, "It's possibly king's move: "+s1+":"+s2
        length = abs(x1-x0)
        assert length>1, "Capture can't be shorter than 2 squares: "+s1+":"+s2
        if length>2:
            assert abs(piece)==3, "Long moves allowed for Kings only: "+s1+":"+s2
        direction = (Move.sign(x1-x0), Move.sign(y1-y0))
        found_capture = None
        for i in range(1,length):
            capture = (x0+i*direction[0], y0+i*direction[1])
            captured = board.get_classic_piece(capture[0], capture[1]) # the captured piece
            if captured==0:
                continue
            assert not (captured!=0 and found_capture), "You can't jump over 2 or more pieces: "+s1+":"+s2
            assert piece * captured < 0, "You can't capture your own piece: "+s1+":"+s2
            found_capture = capture 
        if not found_capture:
            print("Hello")
        assert found_capture, "Illegal notation for simple move. Use hyphen (-) sign: "+s1+":"+s2
        m = Move(x0, y0, x1, y1, found_capture)
        return m
        
    @staticmethod
    def parse_square(s):
        """ Parses square from the given string like a1 => (0,0) """
        x = Board.xx.index(s[0])
        y = int(s[1])-1
        return x, y 
    
    @staticmethod
    def invert_path(s):
        """ Inverts path of the piece by inverting every square in the path """
        if '-' in s:
            d = s.split("-")
            x0, y0 = Move.parse_square(d[0])
            x1, y1 = Move.parse_square(d[1])
            
            x0 = 7 - x0
            y0 = 7 - y0
            x1 = 7 - x1
            y1 = 7 - y1

            str = Move.square(x0, y0)+"-"+Move.square(x1, y1)
        elif ':' in s:
            d = s.split(":")
            x0, y0 = Move.parse_square(d[0])
            x1, y1 = Move.parse_square(d[1])
            
            x0 = 7 - x0
            y0 = 7 - y0
            x1 = 7 - x1
            y1 = 7 - y1
            
            str = Move.square(x0, y0)+":"+Move.square(x1, y1)
            for n in range(2,len(d)):
                x2, y2 = Move.parse_square(d[n])
                
                x2 = 7 - x2
                y2 = 7 - y2
                str += ":"+Move.square(x2, y2)
        return str
    
    @staticmethod
    def avg(a,b):
        """ average of two values """ 
        return int((a+b)/2)
    
    @staticmethod
    def sign(a):
        if a>0:
            return 1
        if a<0:
            return -1
        return 0
     
    @staticmethod
    def parse_action(action):
        """ @see calc_id() """
        # move_id is a bitmask of length 11
        # 11100  11111 1
        # vector from  additional_capture(s)
        #long_capture = action & 1
        #action = action >> 1
        start = action & 31 # start/from
        action = action >> 5
        vector = action
        length = int(vector/4)+1
        direction_id = vector % 4
        direction = Move.__directions[direction_id]
        x0,y0 = (start % 4) * 2, int(start/4)
        if y0 % 2 == 1:
            x0 += 1
        x1,y1 = x0 + direction[0]*length, y0 + direction[1]*length
        assert 0<=x0 and x0<8 and 0<=x1 and x1<8 and 0<=y0 and y0<8 and 0<=y1 and y1<8, "Illegal action: "+str(action)+"=>"+str((x0,y0,x1,y1))   
        m = Move(x0,y0,x1,y1, None)
        # m.is_long_capture = long_capture
        return m
