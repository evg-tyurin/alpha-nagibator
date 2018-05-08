from __future__ import print_function

import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, datetime, os
from Debug import Debug

from checkers.CheckersLogic import Move

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.debug = Debug()
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder = os.path.join("Debug", "Arena", dt+"_"+str(os.getpid()))
        os.makedirs(self.folder)
        self.print_boards = False

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0; currently this is 0.0001
        """

        # reset player's state
        self.player1.reset()
        self.player2.reset()
            
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        lastMove = []
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            #if verbose:
            #    assert(self.display)
            #    print("Turn ", str(it), "Player ", str(curPlayer))
            #    self.display(board)
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            
            if lastMove and not board.last_long_capture: 
                players[curPlayer+1].makeOpponentMove(lastMove)
                del lastMove[:]
            action = players[curPlayer+1].play(canonicalBoard)

            valids = self.game.getValidMoves(canonicalBoard,1)

            if valids[action]==0:
                print(action)
                canonicalBoard.display()
                assert valids[action] >0, "Illegal move: "+str(Move.parse_action(action))
            
            # board.legal_moves = canonicalBoard.legal_moves
            #self.debug.print_legal_moves(board)

            previousBoard = board
            action = canonicalBoard.transform_action_for_board(action, board)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            lastMove.append(board.executed_moves[-1])
            # debug next state
            board.id = previousBoard.id + "-?"
            if board.halfMoves > 100:
                print("info: board.id:", previousBoard.id, "=>", board.id, ", pieces:", board.count_pieces(), ", halfMoves:", board.halfMoves, ", no-progress:", board.noProgressCount)
            if self.print_boards:
                s = self.game.stringRepresentation(previousBoard)
                s1 = self.game.stringRepresentation(board)
                self.debug.print_to_file(board, s1, "Newly generated position, previous one: "+s)
            
            #board.display()
            #input("press enter to continue: ");
            
        result = curPlayer * self.game.getGameEnded(board, curPlayer)
            
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(result))
            self.display(board)
            
        # print game record to a file
        self.game.printGameRecord(board, curPlayer, self.folder)
        
        return result
    
    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            eps += 1
            print("")
            print("Episode ",eps)
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
                
            # bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            eps += 1
            print("")
            print("Episode ",eps)
            gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            
            # bookkeeping + plot progress
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
            
        bar.finish()

        return oneWon, twoWon, draws
