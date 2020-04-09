import numpy as np
from Connect4_OneRow.board import Board

class Connect3OneRowGame:
    """
    A very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 4
        winNumber: 2
    """

    def __init__(self):
        self.columns = 4

    def get_init_board(self):
        b = Board()
        return np.array(b.pieces)

    def get_board_size(self):
        return self.columns

    def get_action_size(self):
        return self.columns

    def get_next_state(self, pieces, player, action):
        b = Board(pieces)
        move = (int(action))
        b.execute_move(move, player)

        # Return the new game, but
        # change the perspective of the game with negative
        return (b.pieces, -player)

    def get_valid_moves(self, pieces, player):
        # All moves are invalid by default
        valid_moves = [0] * self.get_action_size()
        b = Board(pieces)

        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            # No valid moves
            return valid_moves

        # Mark all legal moves
        for legalIndex in legalMoves:
            valid_moves[legalIndex] = 1

        return valid_moves

    def get_game_ended(self, pieces, player):
        # return 0 if not ended, 1 if player 1 wins, -1 if player 1 lost

        b = Board(pieces)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0

        # Draws have very little (but positive reward)
        return 1e-4

    def get_canonical_board(self, pieces, player):
        return player * pieces

    def string_representation(self, pieces):
        b = Board(pieces)
        return b.tostring()
