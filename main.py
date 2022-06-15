import argparse
import os
import torch
from model.image_classifier import  ImageClassifier


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=40, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--model_name', type=str, default='ImageClass', help='name your model')
    parser.add_argument('--transforms', type=str, default=None)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--gpu_mode', type=str2bool, default=True)
    parser.add_argument('--benchmark_mode', type=str2bool, default=True)

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    # declare instance for GAN
    net = ImageClassifier(args)

    if args.is_train:
        net.train()
        print(" [*] Training finished!")

    else:
        net.load()

    # valid model
    print(" [*] Testing finished!")



if __name__ == '__main__':
    main()