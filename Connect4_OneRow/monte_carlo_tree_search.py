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

        # TODO: implement

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps[s]:
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
        next_s = self.game.get_canonical_forms(next_s, next_player)

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










