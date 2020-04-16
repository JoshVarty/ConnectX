import unittest
from Connect2_OneRow.monte_carlo_tree_search import Node, MCTS

class NodeTests(unittest.TestCase):

    def test_initialization(self):
        node = Node(0.5)

        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.prior, 0.5)
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.expanded())
        self.assertEqual(node.value(), 0)

    def test_exploration_noise(self):
        node = Node(0.5)
        node.children = {
            0: Node(0.5),
            1: Node(0.5)
        }

        node.add_exploration_noise(dirichlet_alpha=0.03, exploration_fraction=0.25)

        # Ensure noise changes children
        self.assertNotEqual(node.children[0].prior, 0.5)
        self.assertNotEqual(node.children[1].prior, 0.5)


    def test_selection(self):
        node = Node(0.5)
        c0 = Node(0.5)
        c1 = Node(0.5)
        c2 = Node(0.5)
        node.visit_count = 1
        c0.visit_count = 0
        c2.visit_count = 0
        c2.visit_count = 1

        node.children = {
            0: c0,
            1: c1,
            2: c2,
        }

        action = node.select_action(temperature=0)
        self.assertEqual(action, 2)

    def test_expansion(self):
        node = Node(0.5)

        state = [0, 0, 0, 0]
        action_probs = [0.25, 0.15, 0.5, 0.1]
        to_play = 1

        node.expand(state, to_play, action_probs)

        self.assertEqual(len(node.children), 4)
        self.assertTrue(node.expanded())
        self.assertEqual(node.to_play, to_play)
        self.assertEqual(node.children[0].prior, 0.25)
        self.assertEqual(node.children[1].prior, 0.15)
        self.assertEqual(node.children[2].prior, 0.50)
        self.assertEqual(node.children[3].prior, 0.10)


class MCTSTests(unittest.TestCase):

    def test_ucb_score(self):
        pass

if __name__ == '__main__':
    unittest.main()
