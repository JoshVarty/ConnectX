import os
import numpy as np
from random import shuffle

import torch
import torch.optim as optim

from collections import deque
from Connect2_OneRow.monte_carlo_tree_search import MCTS

class Trainer:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

        self.all = []

    def exceute_episode(self):

        train_examples = []
        board = self.game.get_init_board()
        current_player = 1

        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_board(board, current_player)

            temp = int(episode_step < self.args['tempThreshold'])
            action_probs = self.mcts.get_action_prob(canonical_board, temp=temp)
            train_examples.append((canonical_board, current_player, action_probs, None))

            action = np.random.choice(len(action_probs), p=action_probs)

            board, current_player = self.game.get_next_state(board, current_player, action)

            r = self.game.get_game_ended(board, current_player)

            #TODO: Can we clean this up?
            if r != 0:
                # [Board, currentPlayer, actionProbabilities, None]
                return [(x[0], x[2], r * ((-1) ** (x[1] != current_player))) for x in train_examples]

    def learn(self):

        for i in range(1, self.args['numIters'] + 1):

            train_examples = []

            for eps in range(self.args['numEps']):
                self.mcts = MCTS(self.game, self.model, self.args)
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

                self.all.extend(iteration_train_examples)


            shuffle(train_examples)

            self.train(train_examples)

            self.save_checkpoint(folder=".", filename="latest.pth")


    def train(self, examples):

        optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous().cuda()
                target_pis = target_pis.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0])
            print(target_pis[0])


    def loss_pi(self, targets, outputs):
        # loss_fn = torch.nn.KLDivLoss()
        # return loss_fn(torch.log(outputs), targets)
        # Try cross entropy loss
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):

        x = torch.FloatTensor([0.01]).cuda()
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        loss = torch.max(x, loss)
        return loss


    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)



