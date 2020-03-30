import argparse
import yaml
import logging

from vanilla_ae.train import TrainAE
from self_supervised_ae.train import TrainSelfSupervisedAE


def main():
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file',
                        default='./configs/train_vanilla_ae.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    ae_type = config['ae_type']
    ae_type_to_cls = {'AE': TrainAE,
                      'SelfSupervisedAE': TrainSelfSupervisedAE}
    trainer = ae_type_to_cls[ae_type](config)
    trainer.run_training()


if __name__ == '__main__':
    main()
