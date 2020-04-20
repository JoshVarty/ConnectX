import torch
import math
import numpy as np


def ucb_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    x = value_score + prior_score
    return x


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0

        return self.value_sum / self.visit_count



    def select_action(self, temperature):
        visit_counts = np.array(
            [child.visit_count for child in self.children.values()]
        )
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


    def expand(self, state, to_play, action_probs):
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prob, self.to_play*-1)

    def __repr__(self):
        priot_str = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {}".format(self.state.__str__(), priot_str, self.visit_count)



class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def add_exploration_noise(self, action_probs, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        noise = np.random.dirichlet([dirichlet_alpha] * len(action_probs))
        frac = exploration_fraction
        action_probs = action_probs * (1 - frac) + noise * frac
        return action_probs


    def run(self, model, state, to_play, add_exploration_noise):

        root = Node(0, to_play)

        # The root is a special case. We expand it manually here
        # because we need to add exploration noise to the root
        # https://www.gwern.net/docs/rl/2017-silver.pdf
        # See: Self-Play under Methods
        action_probs, value = model.predict(state)
        if add_exploration_noise:
            action_probs = self.add_exploration_noise(action_probs, dirichlet_alpha=0.5, exploration_fraction=0.25)

        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)

        # EXPAND root
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                oldNode = node
                action, node = node.select_child()

                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, next_player = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, next_player)

            value = self.game.get_game_ended(next_state, next_player)
            if value is None:
                # EXPAND
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, next_player, action_probs)

            self.backpropagate(search_path, value, next_player)

        return root


    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            x = value if node.to_play == to_play else -value
            print(node.to_play, to_play, x)
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

        print()



    def get_action_prob(self, canonicalBoard, numberOfMCTSSimulations=15, temp=1):

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

        # From the paper: We want to explore many nodes from the root node so we add some
        # noise to the probabilities we've collected
        if np.count_nonzero(canonicalBoard) == 0:
            probs = np.array(probs)
            epsilon = 0.25
            alpha = np.ones_like(probs) * 0.03
            probs = (1 - epsilon) * np.array(probs) + epsilon * np.random.dirichlet(alpha=alpha)

        return list(probs)

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
                    u = self.Qsa[(s, a)] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
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










