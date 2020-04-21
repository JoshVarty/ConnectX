import torch
import math
import numpy as np


def ucb_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = -child.value()
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
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), priot_str, self.visit_count, self.value())



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
            action_probs = self.add_exploration_noise(action_probs, dirichlet_alpha=0.3, exploration_fraction=0.25)

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
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_game_ended(next_state, player=1)
            if value is None:
                # EXPAND
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root


    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            x = value if node.to_play == to_play else -value
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1












