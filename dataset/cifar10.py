import pickle
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from dataset.AutoAugment.autoaugment import CIFAR10Policy


class Cifar10Dataset(Dataset):

    def __init__(self, base_path, is_train=True, do_augment=False):
        self.base_path = base_path
        self.is_train = is_train
        self.augment = do_augment
        self.augment_policy = CIFAR10Policy()
        self.data, self.labels, self.cls_names = self.read_data()

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    def read_data(self):
        search = '**/data_batch_*' if self.is_train else '**/test_batch'
        files = glob.glob(os.path.join(self.base_path, search), recursive=True)
        files = sorted(files)
        data = []
        labels = []
        for f in files:
            dataset = self.unpickle(f)
            data += [np.asarray(dataset[b'data'])]
            labels += dataset[b'labels']

        data = np.concatenate(data, axis=0)

        meta_file = glob.glob(os.path.join(self.base_path, '**/*.meta'),
                              recursive=True)[0]
        meta = self.unpickle(meta_file)

        return data, labels, meta[b'label_names']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        r = sample[:1024].reshape((32, 32))
        g = sample[1024:2048].reshape((32, 32))
        b = sample[2048:].reshape((32, 32))
        image = np.dstack((r, g, b))
        if self.augment:
            image = Image.fromarray(image)
            image = self.augment_policy(image)
            image = np.array(image)
        image = image.astype(np.float32)
        image = (image / 255.) - 0.5
        label = self.labels[idx]
        return image, label
