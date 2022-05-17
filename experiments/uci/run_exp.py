import numpy as np
import tensorflow as tf
import copy
import time
import pandas as pd
import sys
import itertools
import os


#import matplotlib.pyplot as plt
sys.path.append("../../")
from svgp_nn_inducing.tf2.kernel.matern import MaternKernel
from svgp_nn_inducing.tf2.kernel.rbf_ard import RBF_ARD
from svgp_nn_inducing.tf2.likelihoods import GaussianLikelihood, BernoulliLikelihood, BernoulliLikelihood_sigmoid, \
    RobustLikelihood
from svgp_nn_inducing.tf2.sgp_vi import SVGP_Titsias, SVGP_NN, SVGP_SOLVE, SWSGP
from svgp_nn_inducing.tf2.utils import get_num_params, save_model, load_model
from svgp_nn_inducing.tf2.callbacks import EpochCSVLogger
from svgp_nn_inducing.tf2.options import options
from svgp_nn_inducing.tf2.dataset_generator import BigDatasetLoader
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)

# tf.config.run_functions_eagerly(True)

#### Exapmle ####

#python --dataset_name kin40k --dataset_nsplit 0 --modelSVGP nn --Ptype reg --scaling robust --ll gauss --nGPU 0 --nip 50 --nhn1 50 --nhl1 1 --rdropout 0.5
#python --dataset_name bank_market --dataset_nsplit 0 --modelSVGP nn --Ptype class --scaling robust --ll bern --nGPU 0 --nip 50 --nhn1 50 --nhl1 1 --rdropout 0.5

######################## inputs ###############################
opt = options().parse()
dataset_name = opt.dataset_name
if int(opt.nGPU) >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    #tf.config.experimental.set_virtual_device_configuration(
     #   gpus[0],
      #  [
       #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)
       # ]
    #)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if opt.Ptype.lower() == 'reg':
    likelihood = GaussianLikelihood()
elif opt.Ptype.lower() == 'class':
    if opt.ll.lower() == 'bern':
        likelihood = BernoulliLikelihood(ngh = 50)
    elif opt.ll.lower() == 'bern_sig':
        likelihood = BernoulliLikelihood_sigmoid(ngh = 50)
    elif opt.ll.lower() == 'robust':
        likelihood = RobustLikelihood()
    else:
        raise Exception('Please define a propser likelihood for classification.')

path_results = "results/{}/SVGP_{}/s{}/".format(opt.dataset_name, opt.modelSVGP.lower().upper(),opt.dataset_nsplit)
if not os.path.exists(path_results):
    os.makedirs(path_results)

if opt.scaling.lower() == 'meanstd':
    scaler = StandardScaler()
elif opt.scaling.lower() == 'minmax':
    scaler = MinMaxScaler(feature_range=(-1, 1))
elif opt.scaling.lower() == 'maxabs':
    scaler = MaxAbsScaler()
elif opt.scaling.lower() == 'robust':
    scaler = RobustScaler()
else:
    scaler = None

if opt.Ptype.lower() == 'reg':
    standardize_output = True
elif opt.Ptype.lower() == 'class':
    standardize_output = False

# Load data
dataset_loader = BigDatasetLoader("../../data/{}/s{}/".format(dataset_name,opt.dataset_nsplit), opt.BatchSize, shuffle=False, standardize_input = True, standardize_output=standardize_output)
data_train = dataset_loader.get_train()
data_test = dataset_loader.get_test()

y_mean = dataset_loader.y_mean
y_std = dataset_loader.y_std

######################## model ###############################
if opt.kernel == 'matern':
    kernel = MaternKernel(length_scale = 0.0,
                    noise_scale = 0.0, output_scale = 0.0, num_dims = dataset_loader.n_attributes)
elif opt.kernel == 'rbf':
    kernel = RBF_ARD(log_lengthscales = [0.0]*dataset_loader.n_attributes,
                    log_sigma0 = 0.0, log_sigma = 0.0)
else:
    raise Exception('not implemented.')
if opt.modelSVGP.lower() == 'titsias':
    epoch = opt.nEpoch
    num_inducing_points = opt.nip
    batch_size = opt.BatchSize
    inducing_points = dataset_loader.sample(num_inducing_points)
    inducing_points_prior = copy.deepcopy(inducing_points)
    model = SVGP_Titsias(kernel, likelihood, inducing_points, dataset_loader.n_train, dataset_loader.n_attributes, y_mean, y_std, path_results= path_results)
    file_name = 'experiment_ll_%s_NIP_%d_batchsize_%d' % (opt.ll.lower(), num_inducing_points, batch_size)
    print('NIP {}, BS {}'.format(num_inducing_points, batch_size))
elif opt.modelSVGP.lower() == 'solve':
    epoch = opt.nEpoch
    num_inducing_points_u = opt.nip
    num_inducing_points_v = opt.nip
    batch_size = opt.BatchSize
    inducing_points = dataset_loader.sample(num_inducing_points_u+num_inducing_points_v)
    inducing_points_prior = copy.deepcopy(inducing_points)
    inducing_points_u = inducing_points[0:num_inducing_points_u]
    inducing_points_v = inducing_points[num_inducing_points_u:]
    model = SVGP_SOLVE(kernel, likelihood, inducing_points_u, inducing_points_v, dataset_loader.n_train, dataset_loader.n_attributes, y_mean, y_std, path_results= path_results)
    file_name = 'experiment_ll_%s_NIPu_%d_v_%d_batchsize_%d' % (opt.ll.lower(),num_inducing_points_u,
                                                             num_inducing_points_v, batch_size)
    print('NIP_h {}, NIP_v, BS {}'.format(num_inducing_points_u, num_inducing_points_v, batch_size))

elif opt.modelSVGP.lower() == 'swsgp':
    epoch = opt.nEpoch
    num_inducing_points = opt.nip
    num_inducing_closest = opt.ncip
    batch_size = opt.BatchSize
    inducing_points = dataset_loader.sample(num_inducing_points)
    model = SWSGP(kernel, likelihood, inducing_points, num_inducing_closest, dataset_loader.n_train, dataset_loader.n_attributes,
                 y_mean, y_std, n_hidden1=15, n_layers1=2, n_hidden2=15, n_layers2=2, path_results='', seed=0)


    file_name = 'experiment_ll_%s_NIP_total_%d_closest_%d_batchsize_%d' % (opt.ll.lower(),num_inducing_points,
                                                             num_inducing_closest, batch_size)
    print('NIP_total {}, NIP_closet, BS {}'.format(num_inducing_points, num_inducing_closest, batch_size))

elif opt.modelSVGP.lower() == 'nn':
    hidden_size_n1 = opt.nhn1
    hidden_layer_n1 = opt.nhl1
    num_inducing_points = opt.nip
    epoch = opt.nEpoch
    batch_size = opt.BatchSize
    model = SVGP_NN(kernel, likelihood, num_inducing_points, dataset_loader.n_train, dataset_loader.n_attributes,
                   y_mean, y_std, hidden_size_n1, hidden_layer_n1, path_results, dropout_rate= opt.rdropout)
    print('H1 {}, L1 {}, NIP {}, BS {}'.format(hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size))
    file_name = 'experiment_ll_%s_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                (opt.ll.lower(),hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)


# Initialize model
model(dataset_loader.sample(batch_size))

num_pars = get_num_params(model.trainable_variables)
model.summary()

np.random.seed(0)

fitting = True
optimizer = tf.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)

if fitting:
    start = time.time()
    model.compile(optimizer=optimizer, run_eagerly=None)
    steps_test = int(np.ceil(dataset_loader.n_test / opt.BatchSize))
    steps_train = int(np.ceil(dataset_loader.n_train / opt.BatchSize))
    logger = EpochCSVLogger(data_train, data_test, opt.BatchSize, steps_train, steps_test, path_results + file_name + ".txt", predict_test=True)
    history = model.fit(data_train,
                        # batch_size=opt.BatchSize,
                        epochs=opt.nEpoch,
                        steps_per_epoch=steps_train,
                        callbacks=[logger], shuffle = False
                        )
    save_model(model, optimizer, path_results)

    end = time.time()

else:
    load_model(model, optimizer, model.path_results, which_epoch='latest')
    model.compile(optimizer=optimizer, run_eagerly=None)




