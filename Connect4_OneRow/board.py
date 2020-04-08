import numpy as np

class Board:
    """
    Board setup for a very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 4
        winNumber: 2
    """

    def __init__(self, board=None):
        """
        Sets up initial board configuration
        """
        self.columns = 4
        self.win = 2

        if board is None:
            self.pieces = [0, 0, 0, 0]
        else:
            self.pieces = np.copy(board)

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, color):

        moves = set()
        for index in range(self.columns):
            if self.pieces[index] == 0:
                moves.add(index)

        return list(moves)

    def has_legal_moves(self):

        for index in range(self.columns):
            if self[index] == 0:
                return True

        return False

    def is_win(self, color):
        count = 0
        for index in range(self.columns):
            if self.pieces[index] == color:
                count = count + 1
            else:
                count = 0

            if count == self.win:
                return True

        return False

    def execute_move(self, move, color):
        assert self.pieces[move] == 0
        self.pieces[move] = color

    def tostring(self):
        result = ""
        result += "["

        for position in self.pieces:
            result += str(position)
            result += ","

        result = result[:-1]
        result += "]"

        return result


