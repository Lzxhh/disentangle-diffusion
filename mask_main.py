import argparse
import traceback
import logging
import yaml
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import numpy as np

from maskdiffusion import MaskDiffusion
from boundarydiffusion import BoundaryDiffusion
from configs.paths_config import HYBRID_MODEL_PATHS

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Mode
    parser.add_argument('--compute_latent', action='store_true')
    parser.add_argument('--generate_image', action='store_true')
    parser.add_argument('--edit_image', action='store_true')
    parser.add_argument('--edit_block', action='store_true')

    # Default
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1006, help='Random seed')
    parser.add_argument('--exp', type=str, default='./runs/', help='Path for saving running related data.')


    # Sampling
    parser.add_argument('--t_0', type=int, default=400, help='Return step in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--n_train_step', type=int, default=6, help='# of steps during generative pross for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative pross for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of varaince of the generative process')
 

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)



    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(">" * 80)

    print("<" * 80)


    runner = MaskDiffusion(args, config)
    # runner = BoundaryDiffusion(args, config)
    try:
        if args.compute_latent:
            runner.compute_latent()
        elif args.generate_image:
            runner.generate_image()
        elif args.edit_image:
            runner.edit_image()
        elif args.edit_block:
            runner.edit_block()
        else:
            print('Choose one mode!')
            raise ValueError
    except Exception:
        logging.error(traceback.format_exc())


    return 0


if __name__ == '__main__':
    sys.exit(main())
