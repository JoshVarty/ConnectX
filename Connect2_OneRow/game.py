import numpy as np

class Connect2Game:
    """
    A very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 4
        winNumber: 2
    """

    def __init__(self):
        self.columns = 4
        self.win = 2

    def get_init_board(self):
        b = np.zeros((self.columns,), dtype=np.int)
        return b

    def get_board_size(self):
        return self.columns

    def get_action_size(self):
        return self.columns

    def get_next_state(self, board, player, action):
        b = np.copy(board)
        b[action] = player

        # Return the new game, but
        # change the perspective of the game with negative
        return (b, -player)

    def has_legal_moves(self, board):
        for index in range(self.columns):
            if board[index] == 0:
                return True
        return False

    def get_legal_moves(self, board):
        moves = set()
        for index in range(self.columns):
            if board[index] == 0:
                moves.add(index)

        return list(moves)

    def get_valid_moves(self, board, player):
        # All moves are invalid by default
        valid_moves = [0] * self.get_action_size()
        b = np.copy(board)

        legalMoves = self.get_legal_moves(b)
        if len(legalMoves) == 0:
            # No valid moves
            return valid_moves

        # Mark all legal moves
        for legalIndex in legalMoves:
            valid_moves[legalIndex] = 1

        return valid_moves

    def is_win(self, board, player):
        count = 0
        for index in range(self.columns):
            if board[index] == player:
                count = count + 1
            else:
                count = 0

            if count == self.win:
                return True

        return False

    def get_game_ended(self, board, player):
        # return 0 if not ended, 1 if player 1 wins, -1 if player 1 lost

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return 0

        # Draws have very little (but positive reward)
        return 1e-4

    def get_canonical_board(self, board, player):
        return player * board

    def string_representation(self, board):

        result = ""
        result += "["

        for position in board:
            result += str(position)
            result += ","

        result = result[:-1]
        result += "]"

        return result




