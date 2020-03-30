from abc import ABC, abstractmethod, abstractproperty
import os
import yaml
import logging
import copy
import torch
from torch.utils.tensorboard import SummaryWriter

from utilities.saver_loader import SaverLoader


class Trainer(ABC):

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
    @abstractmethod
    def model(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def train_dl(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def val_dl(self):
        raise NotImplementedError

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

    @abstractmethod
    def writer_callbacks(self, train_loss, val_loss):
        raise NotImplementedError

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
                data, label = next(iterator)
            except StopIteration:
                iterator = iter(self.train_dl)
                data, label = next(iterator)
            data = data.cuda(self.train_params['device'])
            label = label.cuda(self.train_params['device'])

            train_loss = self.train_step(data, label)

            if step % 1000 == 0:
                val_loss = self.val_step()
                self.writer.add_scalar('Training/loss', train_loss, step)
                self.writer.add_scalar('Validation/loss', val_loss, step)
                self.writer_callbacks(train_loss, val_loss)
                self.saver_loader.periodic_save(step, self.model,
                                                self.optimizer)
                self.saver_loader.save_best_model(step, self.model,
                                                  self.optimizer, val_loss)

    def train_step(self, data, label):
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(data)
        loss = self.model.loss(prediction, label)
        loss.backward()
        self.optimizer.step()

        return loss.data.item()

    @abstractmethod
    def val_step_callback(self, prediction, label):
        raise NotImplementedError

    def val_step(self):
        self.model.eval()
        val_loss = 0
        for data, label in copy.deepcopy(self.train_dl):
            data = data.cuda(self.train_params['device'])
            label = label.cuda(self.train_params['device'])
            prediction = self.model(data)
            loss = self.model.loss(prediction, label)
            val_loss += loss.data.item()

            self.val_step_callback(prediction, label)

        return val_loss / len(self.train_dl)

    def infer(self):
        pass
