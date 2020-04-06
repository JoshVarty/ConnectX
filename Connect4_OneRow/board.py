

class Board :
    """
    Board setup for a very, very simple game of ConnectX in which we have:
        rows: 1
        columns: 5
        winNumber: 3
    """

    def __init__(self):
        """
        Sets up initial board configuration
        """
        self.columns = 5
        self.win = 3
        self.pieces = [0, 0, 0, 0, 0]

    def __getitem(self, index):
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
                return

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


