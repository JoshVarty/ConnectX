import torch

from Connect2_OneRow.game import Connect2Game
from Connect2_OneRow.model import Connect2Model
from Connect2_OneRow.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 64,
    'num_simulations': 50,
    'numIters': 500,
    'numEps': 100,              # Number of full games (episodes) to run during each iteration
    'tempThreshold': 15,        # Number of iterations before we switch temp from 1 to 0
    'numItersForTrainExamplesHistory': 20,
    'epochs': 2,               # Number of epochs of training per iteration
}

game = Connect2Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect2Model(board_size, action_size, device)

trainer = Trainer(game, model, args)
trainer.learn()
