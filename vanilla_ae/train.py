import torch
from torch.utils.data import DataLoader
import torchvision

from utilities.trainer import Trainer
from vanilla_ae.model import VanillaAutoEncoder
from dataset.cifar10 import Cifar10Dataset


class TrainAE(Trainer):

    def __init__(self, config):

        super(TrainAE, self).__init__(config)

    def model(self):
        if self._model is None:
            self._model = VanillaAutoEncoder(**self.model_kwargs)
            self._model = self._model.cuda(self.train_params['device'])

        return self._model

    def train_dl(self):
        if not self._train_dl:
            self._train_dl = DataLoader(Cifar10Dataset(self.base_data_path,
                                                       do_augment=True),
                                        **self.train_dl_params)
        return self._train_dl

    def val_dl(self):
        if not self._val_dl:
            self._val_dl = DataLoader(Cifar10Dataset(self.base_data_path,
                                                     is_train=False),
                                      **self.val_dl_params)
        return self._val_dl

    def writer_callbacks(self, train_loss, val_loss):
        pass

    def val_step_callback(self, prediction, data):
        one_img = data[0, :, :, :]
        one_pred = prediction[0, :, :, :]
        result = torch.stack((one_img, one_pred), dim=0)
        result = result.permute(0, 3, 1, 2)
        result = (result + 0.5) * 255
        result = torchvision.utils.make_grid(result, nrow=1)
        self.writer.add_image('result', result, 0)
