import torch
import math
import numpy as np


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
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

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

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

    def expand(self, state, to_play, action_probs):
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            self.children[a] = Node(prob)



class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args


    def run(self, model, state, to_play, add_exploration_noise):

        root = Node(0)

        # The root is a special case. We expand it manually here
        # because we need to add exploration noise to the root
        # https://www.gwern.net/docs/rl/2017-silver.pdf
        # See: Self-Play under Methods
        # EXPAND
        action_probs, value = model.predict(state)
        valid_moves = self.game.get_valid_moves(state, 1)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)

        # EXPAND root
        root.expand(state, to_play, action_probs)

        if add_exploration_noise:
            root.add_exploration_noise(dirichlet_alpha=0.3, exploration_fraction=0.25)

        for _ in range(self.args['num_simulations']):

            virtual_to_play = to_play
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

                # Players play turn by turn
                virtual_to_play = virtual_to_play * -1

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            next_state, _ = self.game.get_next_state(state, virtual_to_play, action)

            # EXPAND
            action_probs, value = model.predict(next_state)
            valid_moves = self.game.get_valid_moves(next_state, 1)
            action_probs = action_probs * valid_moves  # mask invalid moves
            action_probs /= np.sum(action_probs)
            node.expand(next_state, virtual_to_play, action_probs)

            self.backpropagate(search_path, value, virtual_to_play)

        return root

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        _, action, child = max(
            (self.ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def ucb_score(self, parent, child):
        prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = child.value()
        else:
            value_score = 0

        return value_score + prior_score


    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1











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










