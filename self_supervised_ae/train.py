from torch.utils.data import DataLoader

from utilities.trainer import Trainer
from self_supervised_ae.model import SelfSupervisedAE
from dataset.cifar10 import Cifar10Rotation


class TrainSelfSupervisedAE(Trainer):

    def __init__(self, config):

        super(TrainSelfSupervisedAE, self).__init__(config)

    @property
    def model(self):
        if self._model is None:
            self._model = SelfSupervisedAE(**self.model_kwargs)
            self._model = self._model.cuda(self.train_params['device'])

        return self._model

    @property
    def train_dl(self):
        if not self._train_dl:
            self._train_dl = DataLoader(Cifar10Rotation(self.base_data_path),
                                        **self.train_dl_params)
        return self._train_dl

    @property
    def val_dl(self):
        if not self._val_dl:
            self._val_dl = DataLoader(Cifar10Rotation(self.base_data_path,
                                                      is_train=False),
                                      **self.val_dl_params)
        return self._val_dl

    def writer_callbacks(self, train_loss, val_loss):
        pass

    def val_step_callback(self, prediction, data):
        pass
