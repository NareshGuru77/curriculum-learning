import torch
import torch.nn as nn

from vanilla_ae.model import Encoder


class SelfSupervisedAE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(SelfSupervisedAE, self).__init__()
        self.encoder = Encoder(**kwargs)
        self.fc1 = nn.Linear(kwargs['encoding_size'], kwargs['n_hiddens'])
        self.fc2 = nn.Linear(kwargs['n_hiddens'], 4)

    def forward(self, in_ts):
        in_ts = in_ts.permute(0, 3, 1, 2)
        encoding = self.encoder(in_ts)
        flat_encoding = torch.flatten(encoding, start_dim=1)
        net = self.fc1(flat_encoding)
        predictions = self.fc2(net)

        return predictions

    @staticmethod
    def loss(predictions, ground_truths):
        cross_entropy = nn.CrossEntropyLoss()
        return cross_entropy(predictions, ground_truths)

    def accuracy(self):
        pass
