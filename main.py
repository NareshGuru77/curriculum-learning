import argparse
import yaml
import logging

from vanilla_ae.train import TrainAE


def main():
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file',
                        default='./configs/train.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    trainer = TrainAE(config)
    trainer.run_training()


if __name__ == '__main__':
    main()
