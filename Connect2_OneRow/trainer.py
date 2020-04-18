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
        state = self.game.get_init_board()
        current_player = 1

        episode_step = 0

        while True:
            episode_step += 1

            canonical_board = self.game.get_canonical_board(state, current_player)

            temp = int(episode_step < self.args['tempThreshold'])
            add_exploration_noise = temp > 0

            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(self.model, canonical_board, to_play=1, add_exploration_noise=False)

            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.prior
            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temp)
            if action_probs[action] == 0:
                x = 5;
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_game_ended(state, current_player)

            if reward is not None:
                ret = []
                # [Board, currentPlayer, actionProbabilities, None]
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    t = (hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player)))
                    ret.append(t)
                return ret

    def learn(self):

        for i in range(1, self.args['numIters'] + 1):

            train_examples = []

            for eps in range(self.args['numEps']):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

                self.all.extend(iteration_train_examples)


            shuffle(train_examples)

            self.train(train_examples)

            self.save_checkpoint(folder=".", filename="latest.pth")


    def train(self, examples):

        optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
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



