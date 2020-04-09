import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout

from Connect4_OneRow.board import Board
from Connect4_OneRow.game import Connect3OneRowGame
from Connect4_OneRow.model import Connect4_OneRow_Model
from Connect4_OneRow.trainer import Trainer
from Connect4_OneRow.monte_carlo_tree_search import MCTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display_graph(trainer, model):

    graph = DiGraph()
    color_map = []

    def generateStates(board=None, player=1, prefix=1):
        if board is None:
            board = Board()

        current_state = board.tostring()

        if current_state in lookup:
            state_visit_count = current_state

        if not graph.has_node(current_state):
            if board.is_win(1):
                color_map.append("green")
                graph.add_node(current_state)
                return
            elif board.is_win(-1):
                color_map.append("red")
                graph.add_node(current_state)
                return
            elif not board.has_legal_moves():
                color_map.append("yellow")
                graph.add_node(current_state)
            else:
                color_map.append("gray")
                graph.add_node(current_state)

        if board.is_win(player) or board.is_win(-player):
            return

        valid_moves = board.get_legal_moves(player)

        for move in valid_moves:
            next_board = Board(board.pieces)
            next_board.pieces[move] = player
            next_state = next_board.tostring()

            generateStates(next_board, player * -1)

            graph.add_edge(current_state, next_state)

    # Now that we've got a trained model, let's visualize the tree
    lookup = {}
    for pieces, policy, value in trainer.all:

        if np.count_nonzero(pieces) % 2 == 0:
            board = Board(pieces)
            state_string = board.tostring()

            if state_string not in lookup:
                lookup[state_string] = 0

            lookup[state_string] = lookup[state_string] + 1

        else:
            board = Board(pieces * -1)
            state_string = board.tostring()

            if state_string not in lookup:
                lookup[state_string] = 0

            lookup[state_string] = lookup[state_string] + 1

    generateStates()

    pos = graphviz_layout(graph, prog='dot')

    l = {}
    count_pos = {}
    nx.draw(graph, pos, with_labels=True, arrows=True, node_color=color_map)
    for node in graph.nodes:
        if node in lookup:
            l[node] = str(lookup[node])
        else:
            l[node] = ''

        x, y = pos[node]
        count_pos[node] = (x + 15, y + 10)

    nx.draw_networkx_labels(graph, pos=count_pos, font_size=8, font_color='black', labels=l)

    # Plot actions
    action_lookup = {}

    for node in graph.nodes:

        if l[node] == '':
            continue

        pieces = node[1:-1].split(',')

        arr = np.zeros((4,))
        for index, piece in enumerate(pieces):
            arr[index] = int(piece)

        if np.count_nonzero(arr) % 2 != 0:
            arr = arr * -1

        pi, v = model.predict(arr)
        action_lookup[node] = np.around(pi, decimals=1)

        x, y = pos[node]
        pos[node] = (x, y - 10)

    nx.draw_networkx_labels(graph, pos=pos, font_size=8, font_color='black', labels=action_lookup)

    plt.show()


args = {
    'batch_size': 64,
    'numIters': 300,
    'numEps': 100,              # Number of full games (episodes) to run during each iteration
    'tempThreshold': 15,        # Number of iterations before we switch temp from 1 to 0
    'numItersForTrainExamplesHistory': 20,
    'updateThreshold': 0.6,     # Percentage wins required against previous model required in order to update model
    'epochs': 5,               # Number of epochs of training per iteration
}

game = Connect3OneRowGame()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4_OneRow_Model(board_size, action_size, device)

trainer = Trainer(game, model, args)
trainer.learn()

display_graph(trainer, model)
