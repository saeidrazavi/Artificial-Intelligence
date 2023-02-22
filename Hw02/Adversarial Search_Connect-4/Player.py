from Board import BoardUtility
import random
import numpy as np


class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return random.choice(BoardUtility.get_valid_locations(board))


class HumanPlayer(Player):
    def play(self, board):
        move = int(input("input the next column index 0 to 8:"))
        return move


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=5):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board, counter=0, alpha=-np.inf, beta=np.inf, indicator=True):
        """
        Inputs : 
           board : 7*9 numpy array. 0 for empty cell, 1 and 2 for cells containig a piece.
        return the next move(columns to play in) of the player based on minimax algorithm.
        """
        if (self.depth-counter == 0 or BoardUtility.is_terminal_state(board)):
            return BoardUtility.score_position(board, self.piece)

        if (indicator):
            maxEva = -np.inf
            available_list = BoardUtility.get_valid_locations(board)
            if(self.depth-counter == self.depth):
                dic = {key: [] for key in available_list}
            for child in available_list:
                r = BoardUtility.get_next_open_row(board, child)
                copy_board = np.copy(board)
                copy_board[r, child] = 1 if self.piece == 1 else 2
                eva = self.play(copy_board, counter+1, alpha, beta, False)
                if(self.depth-counter == self.depth):
                    dic[child] = eva
                maxEva = max(maxEva, eva)
                alpha = max(alpha, maxEva)
                if (beta <= alpha):
                    break
            if(self.depth-counter == self.depth):
                move = [k for k, v in dic.items() if v == maxEva][0]
                return move
            return maxEva

        else:
            minEva = +np.inf
            available_list = BoardUtility.get_valid_locations(
                board)
            for child in available_list:
                r = BoardUtility.get_next_open_row(board, child)
                copy_board = np.copy(board)
                copy_board[r, child] = 1 if self.piece == 2 else 2
                eva = self.play(copy_board, counter+1, alpha, beta, True)
                minEva = min(minEva, eva)
                beta = min(beta, eva)
                if (beta <= alpha):
                    break
            return minEva


class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=5, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board, counter=0, alpha=-np.inf, beta=np.inf, indicator=True):
        """
        Inputs : 
           board : 7*9 numpy array. 0 for empty cell, 1 and 2 for cells containig a piece.
        same as above but each time you are playing as max choose a random move instead of the best move
        with probability self.prob_stochastic.
        """
        # Todo: implement minimax algorithm with alpha beta pruning
        if (self.depth-counter == 0 or BoardUtility.is_terminal_state(board)):
            return BoardUtility.score_position(board, self.piece)

        if (indicator):
            maxEva = -np.inf
            available_list = BoardUtility.get_valid_locations(board)
            if(self.depth-counter == self.depth):
                dic = {key: [] for key in available_list}
                random_move = random.choice(
                    BoardUtility.get_valid_locations(board))
            for child in available_list:
                r = BoardUtility.get_next_open_row(board, child)
                copy_board = np.copy(board)
                copy_board[r, child] = 1 if self.piece == 1 else 2
                eva = self.play(copy_board, counter+1, alpha, beta, False)
                if(self.depth-counter == self.depth):
                    dic[child] = eva
                maxEva = max(maxEva, eva)
                alpha = max(alpha, maxEva)
                if (beta <= alpha):
                    break
            if(self.depth-counter == self.depth):
                best_move = [k for k, v in dic.items() if v == maxEva][0]
                random_num = np.random.rand(1)
                return best_move if random_num < self.prob_stochastic else random_move
            return maxEva

        else:
            minEva = +np.inf
            available_list = BoardUtility.get_valid_locations(
                board)
            for child in available_list:
                r = BoardUtility.get_next_open_row(board, child)
                copy_board = np.copy(board)
                copy_board[r, child] = 1 if self.piece == 2 else 2
                eva = self.play(copy_board, counter+1, alpha, beta, True)
                minEva = min(minEva, eva)
                beta = min(beta, eva)
                if (beta <= alpha):
                    break
            return minEva
