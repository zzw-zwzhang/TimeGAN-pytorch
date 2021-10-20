""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        # Inputs for the main function
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # original
        self.parser.add_argument(
            '--data_name',
            choices=['sine', 'stock', 'energy'],
            default='stock',
            type=str)
        self.parser.add_argument(
            '--z_dim',
            help='z or data dimension',
            default=6,
            type=int)
        self.parser.add_argument(
            '--seq_len',
            help='sequence length',
            default=24,
            type=int)
        self.parser.add_argument(
            '--module',
            choices=['gru', 'lstm', 'lstmLN'],
            default='gru',
            type=str)
        self.parser.add_argument(
            '--hidden_dim',
            help='hidden state dimensions (should be optimized)',
            default=24,
            type=int)
        self.parser.add_argument(
            '--num_layer',
            help='number of layers (should be optimized)',
            default=3,
            type=int)
        self.parser.add_argument(
            '--iteration',
            help='Training iterations (should be optimized)',
            default=50000,
            type=int)
        self.parser.add_argument(
            '--batch_size',
            help='the number of samples in mini-batch (should be optimized)',
            default=128,
            type=int)
        self.parser.add_argument(
            '--metric_iteration',
            help='iterations of the metric computation',
            default=10,
            type=int)

        # Add
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--model', type=str, default='TimeGAN', help='chooses which model to use. timegan')

        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')

        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')

        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')

        # Train
        self.parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")

        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        self.parser.add_argument('--w_gamma', type=float, default=1, help='Gamma weight')
        self.parser.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
        self.parser.add_argument('--w_e0', type=float, default=10, help='Encoder loss weight')
        self.parser.add_argument('--w_g', type=float, default=100, help='Generator loss weight.')
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.data_name)
        expr_dir = os.path.join(self.opt.outf, self.opt.name)

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt