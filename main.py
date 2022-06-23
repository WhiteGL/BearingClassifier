import argparse
import torch
from model.image_classifier import ImageClassifier
from model.ts_classifier import TSClassifier
from utils.utils import show_loss_acc


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

    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')
    parser.add_argument('--transforms', type=str, default=None)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--gpu_mode', type=str2bool, default=True)
    parser.add_argument('--benchmark_mode', type=str2bool, default=True)
    parser.add_argument('--model_name', type=str, default='Image')
    parser.add_argument('--test', type=str2bool, default=False)

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
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
    if args.model_name == 'Image':
        net = ImageClassifier(args)
    else:
        net = TSClassifier(args)

    if args.test:
        net.test()
    else:
        model, loss_hist, metric_hist = net.train_val()
        print(" [*] Training finished!")
        show_loss_acc(args.epoch, loss_hist, metric_hist, args.model_name)


if __name__ == '__main__':
    main()