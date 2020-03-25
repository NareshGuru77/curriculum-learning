from torch.utils.data import DataLoader
import torch
import argparse
import yaml
import copy
import os
from torch.utils.tensorboard import SummaryWriter
import logging
import torchvision

from vanilla_ae.model import VanillaAutoEncoder
from dataset.cifar10 import Cifar10Dataset
from utilities.saver_loader import SaverLoader


class Trainer:

    def __init__(self, config):
        self.model_kwargs = config['model_kwargs']
        self.train_params = config['train_params']
        self.base_data_path = config['base_data_path']
        self.train_dl_params = config.get('train_dl_params', {})
        self.val_dl_params = config.get('val_dl_params', {})
        self.save_path = config['save_path']
        self.writer = SummaryWriter(log_dir=os.path.join(
            self.save_path, 'tb_log'))
        self.saver_loader = SaverLoader(self.save_path)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

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
            self._train_dl = DataLoader(Cifar10Dataset(self.base_data_path,
                                                       do_augment=True),
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
        restore_path, completed_steps = self.saver_loader.restore_path()
        if restore_path is not None:
            checkpoint = torch.load(restore_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f'restoring model from path:  {restore_path}')

        iterator = iter(self.train_dl)
        for step in range(completed_steps, self.train_params['steps']):
            try:
                data, _ = next(iterator)
            except StopIteration:
                iterator = iter(self.train_dl)
                data, _ = next(iterator)
            data = data.cuda(self.train_params['device'])

            train_loss = self.train_step(data)

            if step % 1000 == 0:
                self.writer.add_scalar('Loss/train_mse', train_loss, step)
                val_loss = self.val_step()
                self.writer.add_scalar('Loss/val_mse', val_loss, step)

                self.saver_loader.periodic_save(step, self.model,
                                                self.optimizer)
                self.saver_loader.save_best_model(step, self.model,
                                                  self.optimizer, val_loss)

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(data)
        loss = self.model.loss(prediction, data)
        loss.backward()
        self.optimizer.step()

        return loss.data.item()

    def val_step(self):
        self.model.eval()
        val_loss = 0
        for data, _ in copy.deepcopy(self.train_dl):
            data = data.cuda(self.train_params['device'])
            one_img = data[0, :, :, :]
            prediction = self.model(data)
            one_pred = prediction[0, :, :, :]
            result = torch.stack((one_img, one_pred), dim=0)
            result = result.permute(0, 3, 1, 2)
            result = (result + 0.5) * 255
            result = torchvision.utils.make_grid(result, nrow=1)
            self.writer.add_image('result', result, 0)
            loss = self.model.loss(prediction, data)
            val_loss += loss.data.item()

        return val_loss / len(self.train_dl)


def main():
    logging.getLogger().setLevel(logging.INFO)
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
