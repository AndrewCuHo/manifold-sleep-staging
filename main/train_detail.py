import argparse
import os
import utils as util


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class train_detail():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('--train_path', default='',
                                  type=str, help='train dataset path')
        self._parser.add_argument('--IF_TRAIN', type=str2bool, default=True, help='if test')
        self._parser.add_argument('--IF_GPU', type=str2bool, default=True, help='use of GPU')
        self._parser.add_argument('--model', default='resnest50', type=str,
                                  help='')
        self._parser.add_argument('--checkpoints', type=str, default='result_model', help='your checkpoints model name')
        self._parser.add_argument('--loss', type=str, default='CB_loss',
                                  help='')
        self._parser.add_argument('--num_classes', type=int, default='5')
        self._parser.add_argument('--input_height', type=int, default='50')
        self._parser.add_argument('--input_weight', type=int, default='75')
        self._parser.add_argument('--crop_size', type=int, default='100')
        self._parser.add_argument('--batch_size', type=int, default='16')
        self._parser.add_argument('--num_epochs', type=int, default='1000')
        self._parser.add_argument('--init_lr', type=float, default='0.001')
        self._parser.add_argument('--lr_scheduler', type=str, default='cosin')
        self._parser.add_argument('--step_size', type=int, default='100')
        self._parser.add_argument('--multiplier', type=float, default='10')
        self._parser.add_argument('--total_epoch', type=int, default='50')
        self._parser.add_argument('--alpha', type=float, default='0.75')
        self._parser.add_argument('--gamma', type=int, default='2')
        self._parser.add_argument('--alpha', type=float, default='0.6')
        self._parser.add_argument('--gamma', type=int, default='5')
        self._parser.add_argument('--manualSeed', type=int, default=110)
        self._parser.add_argument('--out', default='./log')
        self._parser.add_argument('--UnlabeledPercent', type=int, default=50)
        self._parser.add_argument('--Distrib_Threshold', type=float, default=0.95)
        self._parser.add_argument('--Balance_loss', type=float, default=10)
        self._parser.add_argument('--re', type=str2bool, default=False)
        self._parser.add_argument('--nowEpoch', type=int, default=0)
        self._parser.add_argument('--SelectData', type=int, default=1)
        self._parser.add_argument('--Balance_loss', type=float, default=1)
        self._parser.add_argument('--re', type=str2bool, default=False)
        self._parser.add_argument('--nowEpoch', type=int, default=0)

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        args = vars(self._opt)
        return self._opt
