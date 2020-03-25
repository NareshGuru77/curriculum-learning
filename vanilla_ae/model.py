import torch
import torch.nn as nn


class Block(torch.nn.Module):
    def __init__(self, op, op_params, **kwargs):
        super(Block, self).__init__()

        common_params = {'kernel_size': 3, 'padding': 1}
        self.conv1 = nn.Conv2d(*kwargs['op1'], **common_params)
        self.bn1 = nn.BatchNorm2d(kwargs['op1'][1])
        self.op2 = op(*kwargs['op2'], **common_params, stride=2,
                      **op_params)
        self.bn2 = nn.BatchNorm2d(kwargs['op2'][1])
        self.conv3 = nn.Conv2d(*kwargs['op3'], **common_params)
        self.bn3 = nn.BatchNorm2d(kwargs['op3'][1])

    def forward(self, in_ts):
        net = self.bn1(nn.functional.relu(self.conv1(in_ts)))
        net = self.bn2(nn.functional.relu(self.op2(net)))
        net = self.bn3(nn.functional.relu(self.conv3(net)))

        return net


class Encoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        self.block1 = Block(nn.Conv2d, {}, **kwargs['conv_block1'])
        self.block2 = Block(nn.Conv2d, {}, **kwargs['conv_block2'])

    def forward(self, in_ts):
        net = self.block1(in_ts)
        net = self.block2(net)

        return net


class Decoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.block1 = Block(nn.ConvTranspose2d, {'output_padding': 1},
                            **kwargs['tran_conv_block1'])
        self.block2 = Block(nn.ConvTranspose2d, {'output_padding': 1},
                            **kwargs['tran_conv_block2'])

    def forward(self, in_ts):
        net = self.block1(in_ts)
        net = self.block2(net)

        return net


class VanillaAutoEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VanillaAutoEncoder, self).__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, in_ts):
        in_ts = in_ts.permute(0, 3, 1, 2)
        encoding = self.encoder(in_ts)
        predictions = self.decoder(encoding)
        predictions = predictions.permute(0, 2, 3, 1)

        return predictions

    def loss(self, predictions, ground_truths):
        mse = nn.MSELoss()
        return mse(predictions, ground_truths)
