import numpy as np

ROWS = 7
COLS = 9


class BoardUtility:
    @staticmethod
    def make_move(game_board, col, piece):
        """
        make a new move on the board
        row & col: row and column of the new move
        piece: 1 for first player. 2 for second player
        """
        row = BoardUtility.get_next_open_row(game_board, col)
        assert game_board[row][col] == 0
        game_board[row][col] = piece

    @staticmethod
    def is_column_full(game_board, col):
        return game_board[0][col] != 0

    # --------------------------

    def horiz_score(game_board, piece, n):
        count = 0
        for c in range(COLS - n + 1):
            for r in range(ROWS):
                if (np.array_equal(np.array(game_board[r,c:c+n]).reshape(-1),np.array([piece for _ in range(n)]).reshape(-1))):
                    count += 1
        return count
    # --------------------------

    def vert_score(game_board, piece, n):
        count = 0
        # print(game_board)
        for c in range(COLS):
            for r in range(ROWS - n + 1):
                # print(f"original{game_board[r][c]}")
                # print(f":mine{game_board[r:r+n][c]}")
                # print(f":third{game_board[r:r+n,c]}")
                # print(f"1:{np.array(game_board[r:r+n,c]).reshape(-1)}")
                # print(f"2:{np.array([piece for _ in range(n)]).reshape(-1)}")
                if (np.array_equal(np.array(game_board[r:r+n,c]).reshape(-1),np.array([piece for _ in range(n)]).reshape(-1))):
                    count += 1

        return count

    # ---------------------------

    def diag_score(game_board, piece, n):
        count = 0

        for c in range(COLS - n+1):
            for r in range(n-1, ROWS):
                board_game_status = np.array(
                    [game_board[r-i,c+i] for i in range(n)]).reshape(-1)
                if(np.array_equal(board_game_status ,np.array([piece for _ in range(n)]).reshape(-1))):
                    count += 1

        for c in range(n-1, COLS):
            for r in range(n-1, ROWS):
                board_game_status = np.array(
                    [game_board[r-i,c-i] for i in range(n)]).reshape(-1)
                if(np.array_equal(board_game_status, np.array([piece for _ in range(n)]).reshape(-1))):
                    count += 1

        return count

    @staticmethod
    def get_next_open_row(game_board, col):
        """
        returns the first empty row in a column from the top.
        useful for knowing where the piece will fall if this
        column is played.
        """
        for r in range(ROWS - 1, -1, -1):
            if game_board[r][col] == 0:
                return r

    @staticmethod
    def has_player_won(game_board, player_piece):
        """
        piece:  1 or 2.
        return: True if the player with the input piece has won.
                False if the player with the input piece has not won.
        """
        # checking horizontally
        for c in range(COLS - 3):
            for r in range(ROWS):
                if game_board[r][c] == player_piece and game_board[r][c + 1] == player_piece and game_board[r][
                        c + 2] == player_piece and game_board[r][c + 3] == player_piece:
                    return True

        # checking vertically
        for c in range(COLS):
            for r in range(ROWS - 3):
                if game_board[r][c] == player_piece and game_board[r + 1][c] == player_piece and game_board[r + 2][
                        c] == player_piece and game_board[r + 3][c] == player_piece:
                    return True

        # checking diagonally
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if game_board[r][c] == player_piece and game_board[r - 1][c + 1] == player_piece and game_board[r - 2][
                    c + 2] == player_piece and \
                        game_board[r - 3][c + 3] == player_piece:
                    return True

        # checking diagonally
        for c in range(3, COLS):
            for r in range(3, ROWS):
                if game_board[r][c] == player_piece and game_board[r - 1][c - 1] == player_piece and game_board[r - 2][
                    c - 2] == player_piece and \
                        game_board[r - 3][c - 3] == player_piece:
                    return True

        return False

    @staticmethod
    def is_draw(game_board):
        return not np.any(game_board == 0)

    @staticmethod
    def score_position(game_board, piece):
        """
        compute the game board score for a given piece.
        you can change this function to use a better heuristic for improvement.
        """
        # score = 0
        # if BoardUtility.has_player_won(game_board, piece):
        #     return 100_000_000_000  # player has won the game give very large score
        # if BoardUtility.has_player_won(game_board, 1 if piece == 2 else 2):
        #     return -100_000_000_000  # player has lost the game give very large negative score

        vertical_score_piece = BoardUtility.vert_score(game_board, piece, 4)*100+BoardUtility.vert_score(
            game_board, piece, 3)*10+BoardUtility.vert_score(game_board, piece, 2)*5
        horizental_score_piece = BoardUtility.horiz_score(game_board, piece, 4)*100+BoardUtility.horiz_score(
            game_board, piece, 3)*10+BoardUtility.horiz_score(game_board, piece, 2)*5
        diagonal_score_piece = BoardUtility.diag_score(game_board, piece, 4)*100+BoardUtility.diag_score(
            game_board, piece, 3)*10+BoardUtility.diag_score(game_board, piece, 2)*5
        piece_total_score = vertical_score_piece + \
            horizental_score_piece+diagonal_score_piece

        opponent = 1 if piece == 2 else 2

        vertical_score_opponent = BoardUtility.vert_score(game_board, opponent, 4)*100+BoardUtility.vert_score(
            game_board, opponent, 3)*10+BoardUtility.vert_score(game_board, opponent, 2)*5
        horizental_score_opponent = BoardUtility.horiz_score(game_board, opponent, 4)*100+BoardUtility.horiz_score(
            game_board, opponent, 3)*10+BoardUtility.horiz_score(game_board, opponent, 2)*5
        diagonal_score_opponent = BoardUtility.diag_score(game_board, opponent, 4)*100+BoardUtility.diag_score(
            game_board, opponent, 3)*10+BoardUtility.diag_score(game_board, opponent, 2)*5
        opponent_score = vertical_score_opponent + \
            horizental_score_opponent+diagonal_score_opponent

        score = piece_total_score-opponent_score
        # todo score the game board based on a heuristic.
        return score
        # return score

    @staticmethod
    def get_valid_locations(game_board):
        """
        returns all the valid columns to make a move.
        """
        valid_locations = []

        for column in range(COLS):
            if not BoardUtility.is_column_full(game_board, column):
                valid_locations.append(column)

        return valid_locations

    @staticmethod
    def is_terminal_state(game_board):
        """
        return True if either of the player have won the game or we have a draw.
        """
        return BoardUtility.has_player_won(game_board, 1) or BoardUtility.has_player_won(game_board,
                                                                                         2) or BoardUtility.is_draw(
            game_board)
