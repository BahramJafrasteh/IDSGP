import argparse
import os

class options():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.parser = parser


    def initialize(self, parser):
        parser.add_argument('--dataset_name', required=True,
                            help='name of the data set (should have subfolders with the same name)')
        parser.add_argument('--scaling', type=str, default='MeanStd',help='scaling method [MeanStd|MinMax|MaxAbs|Robust|None]')
        parser.add_argument('--dataset_nsplit', type=int, default=0,
                            help='data set split number [0|1|2|etc]')
        parser.add_argument('--modelSVGP', type=str, default='nn',
                            help='the available SVGP models are [nn|solve|swsgp|titsias]')
        parser.add_argument('--Ptype', type=str, default='reg',
                            help='Problem type (regression or classification) [reg|class]')
        parser.add_argument('--nGPU', type=int, default=-1,
                            help='GPU number (for cpu use -1) [-1|0|1|2]')
        parser.add_argument('--nEpoch', type=int, default=100,
                            help='Maximum number of epochs (default: 100)')
        parser.add_argument('--BatchSize', type=int, default=100,
                            help='Batch size (default: 100)')
        parser.add_argument('--nip', type=int, default=1024,
                            help='number of inducing points (default: 100)')
        parser.add_argument('--ncip', type=int, default=50,
                            help='number of the closest inducing points (SWSGP)')
        parser.add_argument('--nhn1', type=int, default=5,
                            help='number of hidden nodes for the first neural network (default: 5)')
        parser.add_argument('--nhl1', type=int, default=1,
                            help='number of hidden layers for the first neural network(default: 1)')
        parser.add_argument('--ll', type=str, default='gauss',
                            help='Likelihood type [gauss (Gaussian)|bern (Bernoulli probit)|bern_sig (Bernoulli logit)|'
                                 'robust]')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate ')
        parser.add_argument('--b1', type=float, default=0.9,
                            help='beta 1')
        parser.add_argument('--b2', type=float, default=0.999,
                            help='beta 2')
        parser.add_argument('--rdropout', type=float, default=0.0,
                            help='Dropout rate')
        parser.add_argument('--kernel', type=str, default='matern', help='Kernel type [matern|rbf]')
        return parser
    def parse(self):
        opt = self.parser.parse_args()
        args = opt._get_kwargs()
        path_config = "results/{}/SVGP_{}/s{}/".format(opt.dataset_name, opt.modelSVGP.upper(),opt.dataset_nsplit)
        if not os.path.exists(path_config):
            os.makedirs(path_config)

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = '\t[default: {}] \t[help: {}]\t '.format(str(default), str(self.parser._option_string_actions['--'+k].help))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        with open(path_config+'conf.txt', 'w') as file:
            file.write(message)
        self.opt = opt
        return opt
