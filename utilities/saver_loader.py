import os
import logging
import glob
import torch


class SaverLoader:

    def __init__(self, save_dir, best_name='best', latest_name='latest',
                 models_to_keep=5, save_steps=1000):
        self.save_dir = save_dir
        self.best_name = best_name
        self.latest_name = latest_name
        self.models_to_keep = models_to_keep
        self.save_steps = save_steps
        self.lowest_loss = 1000
        self.sort_key = lambda k: \
            int(k.split('/')[-1].split('-')[-1].split('.')[0])

    def read_models(self):
        best_pths = glob.glob(os.path.join(self.save_dir,
                                           f'{self.best_name}*'))
        best_pths = sorted(best_pths, key=self.sort_key)
        latest_pths = glob.glob(os.path.join(self.save_dir,
                                             f'{self.latest_name}*'))
        latest_pths = sorted(latest_pths, key=self.sort_key)

        return best_pths, latest_pths

    def restore_path(self, prefer_best=False):
        best_pths, latest_pths = self.read_models()

        if prefer_best and len(best_pths) > 0:
            logging.info('returing the path of best model')
            return best_pths[-1], self.sort_key(best_pths[-1])
        elif len(latest_pths) > 0:
            logging.info('returing the path of latest model')
            return latest_pths[-1], self.sort_key(latest_pths[-1])
        else:
            return None, 0

    def delete_old(self, pths, msg):
        if len(pths) > self.models_to_keep:
            logging.info(f'{msg}.. %s', pths[0])
            os.remove(pths[0])

    def save_best_model(self, step, model, optimizer, loss):
        if loss < self.lowest_loss:
            logging.info('saving best model at step: %s', step)
            path = os.path.join(self.save_dir,
                                f'{self.best_name}-%06d.pth' % step)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)
            self.lowest_loss = loss

            best_pths, _ = self.read_models()
            self.delete_old(best_pths, 'deleting old best model')

    def periodic_save(self, step, model, optimizer):
        if step % self.save_steps == 0:
            logging.info('saving model at step: %s', step)
            path = os.path.join(self.save_dir,
                                f'{self.latest_name}-%06d.pth' % step)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)

            _, latest_zips = self.read_models()
            self.delete_old(latest_zips, 'deleting old model')
