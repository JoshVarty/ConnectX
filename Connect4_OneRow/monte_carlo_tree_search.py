import math
import numpy as np


class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.EPS = 1e-8

        self.Qsa = {}       # lookup from (state, action) -> Q value
        self.Nsa = {}       # lookup from (state, action) -> number of times visited
        self.Ns = {}        # lookup from (state) -> number of times visited
        self.Ps = {}        # stores policy decisions returned by neural network

        self.Es = {}        # stores whether the game is ended at given state
        self.Vs = {}        # stores valid moves for a given state

    def get_action_prob(self, canonicalBoard, numberOfMCTSSimulations=10, temp=1):

        for i in range(numberOfMCTSSimulations):
            self.search(canonicalBoard)

        s = self.game.string_representation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        s = self.game.string_representation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonicalBoard, 1)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node, EXPAND
            self.Ps[s], v = self.model.predict(canonicalBoard)
            valid_moves = self.game.get_valid_moves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valid_moves   # mask invalid moves
            self.Ps[s] /= np.sum(self.Ps[s])        # renormalize predictions

            self.Vs[s] = valid_moves
            self.Ns[s] = 0
            return -v

        valid_moves = self.Vs[s]
        current_best = -float('inf')
        best_action = -1

        # pick the action with the highest upper confidence bound?
        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    # We've never taken this action in this state before
                    u = self.Ps[s][a] * math.sqrt(self.Ns[s] + self.EPS)

                if u > current_best:
                    current_best = u
                    best_action = a

        a = best_action
        next_s, next_player = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_board(next_s, next_player)

        v = self.search(next_s)

        # Backup
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v










