from torch.utils.data import DataLoader
import torch
import argparse
import yaml
import copy

from vanilla_ae.model import VanillaAutoEncoder
from dataset.cifar10 import Cifar10Dataset


class Trainer:

    def __init__(self, config):
        self.model_kwargs = config['model_kwargs']
        self.train_params = config['train_params']
        self.base_data_path = config['base_data_path']
        self.train_dl_params = config.get('train_dl_params', {})
        self.val_dl_params = config.get('val_dl_params', {})

        self._model = None
        self._optimizer = None
        self._train_dl = None
        self._val_dl = None

    @property
    def model(self):
        if self._model is None:
            self._model = VanillaAutoEncoder(**self.model_kwargs)
            self._model = self._model.cuda(self.train_params['device'])

        return self._model

    @property
    def train_dl(self):
        if not self._train_dl:
            self._train_dl = DataLoader(Cifar10Dataset(self.base_data_path),
                                        **self.train_dl_params)
        return self._train_dl

    @property
    def val_dl(self):
        if not self._val_dl:
            self._val_dl = DataLoader(Cifar10Dataset(self.base_data_path,
                                                     is_train=False),
                                      **self.val_dl_params)
        return self._val_dl

    @property
    def optimizer(self):
        if not self._optimizer:
            optimizer_params = self.train_params['optimizer']
            optimizer_name = optimizer_params.pop('name')
            try:
                optimizer_fn = getattr(torch.optim, optimizer_name)
            except AttributeError:
                raise ValueError(f'unknown optimizer: {optimizer_name}')

            self._optimizer = optimizer_fn(self.model.parameters(),
                                           **optimizer_params)

        return self._optimizer

    def run_training(self):
        iterator = iter(self.train_dl)
        for i in range(self.train_params['steps']):
            try:
                data, _ = next(iterator)
            except StopIteration:
                iterator = iter(self.train_dl)
                data, _ = next(iterator)
            data = data.cuda(self.train_params['device'])
            self.model.train()
            self.optimizer.zero_grad()
            prediction = self.model(data)
            loss = self.model.loss(prediction, data)
            loss.backward()
            self.optimizer.step()
            if i % 1000 == 0:
                print(f'Training loss: {loss.data.item()}')
                self.model.eval()
                val_loss = 0
                for data, _ in copy.deepcopy(self.train_dl):
                    data = data.cuda(self.train_params['device'])
                    prediction = self.model(data)
                    loss = self.model.loss(prediction, data)
                    val_loss += loss.data.item()

                print(f'Validation loss: {val_loss / len(self.train_dl)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file',
                        default='./configs/train.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    trainer = Trainer(config)
    trainer.run_training()


if __name__ == '__main__':
    main()
