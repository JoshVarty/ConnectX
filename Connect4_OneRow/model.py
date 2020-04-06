import torch.nn as nn
import torch.nn.functional as F

class Connect4_OneRow_Model(nn.Module):

    def __init__(self, game):

        super(Connect4_OneRow_Model, self).__init__()

        self.size = game.get_board_size()
        self.action_size = game.get_action_size()

        self.fc1 = nn.Linear(in_features=self.size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)

        # Two heads on our network
        self.action_head = nn.Linear(in_features=16, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=16, out_features=1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.log_softmax(action_logits, dim=1), F.tanh(value_logit)
