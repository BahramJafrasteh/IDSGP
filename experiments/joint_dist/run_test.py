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
from svgp_nn_inducing.tf2.sgp_vi import SVGP_NN, SVGP_SOLVE, SWSGP, SWSGPU, FullGP, SVGP_Titsias, SVGP_Hensman, SVGP_NNJ, SVGP_NNU
from svgp_nn_inducing.tf2.utils import get_num_params, save_model, load_model
from svgp_nn_inducing.tf2.callbacks import EpochCSVLogger, TimerStopper
from svgp_nn_inducing.tf2.options import options
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)

# tf.config.run_functions_eagerly(True)

#### Example ####

#python --dataset_name kin40k --dataset_nsplit 0 --modelSVGP nn --Ptype reg --scaling robust --ll gauss --nGPU 0 --nip 50 --nhn1 50 --nhl1 1 --rdropout 0.5
#python --dataset_name bank_market --dataset_nsplit 0 --modelSVGP nn --Ptype class --scaling robust --ll bern --nGPU 0 --nip 50 --nhn1 50 --nhl1 1 --rdropout 0.5

######################## inputs ###############################
opt = options().parse()
dataset_name = opt.dataset_name
if int(opt.nGPU) >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.nGPU)
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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


df_train = pd.read_csv("../../data/{}/s{}/{}_train_data.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name), header=None)
df_train_lables = pd.read_csv("../../data/{}/s{}/{}_train_labels.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name),
                              header=None)
df_test = pd.read_csv("../../data/{}/s{}/{}_test_data.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name), header=None)
df_test_labels = pd.read_csv("../../data/{}/s{}/{}_test_labels.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name), header=None)
X_train, y_train = np.array(df_train), np.array(df_train_lables).squeeze()
X_test, y_test = np.array(df_test), np.array(df_test_labels).squeeze()

# Make number of instances divisible by batch size

n_train = X_train .shape[0] // opt.BatchSize * opt.BatchSize
n_test = X_test.shape[0] // opt.BatchSize * opt.BatchSize

X_test = X_test[0:n_test, :]
y_test = y_test[0:n_test]
X_train = X_train[0:n_train, :]
y_train = y_train[0:n_train]

y_mean = 0
y_std = 1

if opt.Ptype.lower() == 'reg':
    standardize_output = True
    if standardize_output:
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
elif opt.Ptype.lower() == 'class':
    if min(y_train) != -1 or max(y_train) != 1:
        raise NotImplementedError("For binary classification the outputs should be -1 or 1")

if opt.scaling.lower() == 'meanstd':
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
elif opt.scaling.lower() == 'minmax':
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
elif opt.scaling.lower() == 'maxabs':
    scaler = MaxAbsScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
elif opt.scaling.lower() == 'robust':
    scaler = RobustScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)


######################## model ###############################
if opt.kernel == 'matern':
    kernel = MaternKernel(length_scale = 0.0,
                    noise_scale = 0.0, output_scale = 0.0, num_dims = X_train.shape[1], nu = 2.5)
elif opt.kernel == 'rbf':
    kernel = RBF_ARD(log_lengthscales = [0.0]*X_train.shape[1],
                    log_sigma0 = 0.0, log_sigma = 0.0)
else:
    raise Exception('not implemented.')
if opt.modelSVGP.lower() == 'titsias':
    epoch = opt.nEpoch
    num_inducing_points = opt.nip
    batch_size = opt.BatchSize
    batch_size_test = batch_size
    shuffle = True
    inducing_points = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_points), :].astype(
        np.float64)
    
    model = SVGP_Hensman(kernel, likelihood, inducing_points, X_train.shape[ 0 ], X_train.shape[1], y_mean, y_std, path_results= path_results)
    file_name = 'experiment_ll_%s_NIP_%d_batchsize_%d' % (opt.ll.lower(), num_inducing_points, batch_size)
    print('NIP {}, BS {}'.format(num_inducing_points, batch_size))
elif opt.modelSVGP.lower() == "full_gp":
    opt.nip = X_train.shape[0]
    opt.BatchSize = X_train.shape[0]
    epoch = opt.nEpoch
    num_inducing_points = opt.nip
    batch_size = opt.BatchSize
    batch_size_test = X_test.shape[0]
    shuffle = True
    inducing_points = X_train
    model = FullGP(kernel, likelihood, X_train.shape[ 0 ], X_train, y_train, y_mean, y_std, path_results=path_results)
    file_name = 'experiment_ll_%s_NIP_%d_batchsize_%d' % (opt.ll.lower(), num_inducing_points, batch_size)
    print('NIP {}, BS {}'.format(num_inducing_points, batch_size))
elif opt.modelSVGP.lower() == 'nn_orig':
    hidden_size_n1 = opt.nhn1
    hidden_layer_n1 = opt.nhl1
    num_inducing_points = opt.nip
    epoch = opt.nEpoch
    batch_size = opt.BatchSize
    batch_size_test = batch_size
    shuffle = True
    model = SVGP_NN(kernel, likelihood, num_inducing_points, X_train.shape[0], X_train.shape[1],
                   y_mean, y_std, hidden_size_n1, hidden_layer_n1, path_results, dropout_rate= opt.rdropout)
    print('H1 {}, L1 {}, NIP {}, BS {}'.format(hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size))
    file_name = 'experiment_ll_%s_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                (opt.ll.lower(),hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)
elif opt.modelSVGP.lower() == 'nnu':
    hidden_size_n1 = opt.nhn1
    hidden_layer_n1 = opt.nhl1
    num_inducing_points = opt.nip
    epoch = opt.nEpoch
    batch_size = opt.BatchSize
    batch_size_test = batch_size
    shuffle = True
    model = SVGP_NNU(kernel, likelihood, num_inducing_points, X_train.shape[0], opt.BatchSize, X_train.shape[1],
                   y_mean, y_std, hidden_size_n1, hidden_layer_n1, path_results, dropout_rate= opt.rdropout)
    print('H1 {}, L1 {}, NIP {}, BS {}'.format(hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size))
    file_name = 'experiment_ll_%s_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                (opt.ll.lower(),hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)
elif opt.modelSVGP.lower() == 'nnj':
    hidden_size_n1 = opt.nhn1
    hidden_layer_n1 = opt.nhl1
    num_inducing_points = opt.nip
    epoch = opt.nEpoch
    batch_size = opt.BatchSize
    batch_size_test = batch_size
    shuffle = True
    model = SVGP_NNJ(kernel, likelihood, num_inducing_points, X_train.shape[0], opt.BatchSize, X_train.shape[1],
                   y_mean, y_std, hidden_size_n1, hidden_layer_n1, path_results, dropout_rate= opt.rdropout)
    print('H1 {}, L1 {}, NIP {}, BS {}'.format(hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size))
    file_name = 'experiment_ll_%s_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                (opt.ll.lower(),hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)
elif opt.modelSVGP.lower() == 'solve':
    num_inducing_points_u = opt.nip
    num_inducing_points_v = opt.nip
    epoch = opt.nEpoch
    batch_size = opt.BatchSize
    batch_size_test = batch_size
    shuffle = True
    inducing_points = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_points_u+num_inducing_points_v), :].astype(
        np.float64)
    inducing_points_prior = copy.deepcopy(inducing_points)
    inducing_points_u = inducing_points[0:num_inducing_points_u]
    inducing_points_v = inducing_points[num_inducing_points_u:]
    model = SVGP_SOLVE(kernel, likelihood, inducing_points_u, inducing_points_v, X_train.shape[ 0 ], X_train.shape[1], y_mean, y_std, path_results= path_results)
    file_name = 'experiment_ll_%s_NIPu_%d_v_%d_batchsize_%d' % (opt.ll.lower(),num_inducing_points_u,
                                                             num_inducing_points_v, batch_size)
    print('NIP_h {}, NIP_v, BS {}'.format(num_inducing_points_u, num_inducing_points_v, batch_size))
elif opt.modelSVGP.lower() == 'swsgpu':
    epoch = opt.nEpoch
    num_inducing_points = opt.nip
    num_inducing_closest = int(opt.ncip / opt.BatchSize)
    batch_size = opt.BatchSize
    batch_size_test = batch_size
    shuffle = True
    inducing_points = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_points), :].astype(
        np.float64)
    model = SWSGPU(kernel, likelihood, inducing_points, num_inducing_closest, X_train.shape[ 0 ], X_train.shape[1],
                 y_mean, y_std, n_hidden1=15, n_layers1=2, n_hidden2=15, n_layers2=2, path_results='', seed=0)


    file_name = 'experiment_ll_%s_NIP_total_%d_closest_%d_batchsize_%d' % (opt.ll.lower(),num_inducing_points,
                                                             num_inducing_closest, batch_size)
    print('NIP_total {}, NIP_closest {}, BS {}'.format(num_inducing_points, num_inducing_closest, batch_size))


# Initialize model

model(X_train[0:opt.BatchSize])

num_pars = get_num_params(model.trainable_variables)
model.summary()
optimizer = tf.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)


np.random.seed(0)

fitting = False


if fitting:

    start = time.time()
    
    model.compile(optimizer=optimizer, run_eagerly=None)

    steps_test = int(np.ceil(X_test.shape[0] / batch_size_test))
    steps_train = int(np.ceil(X_train.shape[0] / opt.BatchSize))

    # Max training time 1 dat -> 86400s
#    logger2 = TimerStopper((X_train, y_train), (X_test, y_test), opt.BatchSize, batch_size_test,
#                            steps_train, steps_test, filename = path_results + file_name + ".txt",
#                            max_seconds = 86400, path_results=path_results)
    logger = EpochCSVLogger((X_train, y_train), (X_test, y_test), opt.BatchSize, steps_train, steps_test, path_results + file_name + ".txt", predict_test=True)
    history = model.fit(X_train, y_train,
                        batch_size=opt.BatchSize,
                        epochs=200,
                        steps_per_epoch = steps_train,
                        callbacks=[logger], 
                        shuffle = shuffle
                        )

    end = time.time()

    path_results = "results/{}/SVGP_{}/s{}/batchsize{}_M{}/".format(opt.dataset_name, opt.modelSVGP.lower().upper(),opt.dataset_nsplit, opt.BatchSize, opt.nip)
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    save_model(model, optimizer, path_results)
    
else:
    path_results = "results/{}/SVGP_{}/s{}/batchsize{}_M{}/".format(opt.dataset_name, opt.modelSVGP.lower().upper(),opt.dataset_nsplit, opt.BatchSize, opt.nip)
    model.compile(optimizer=optimizer, run_eagerly=None)
    load_model(model, None, path_results, which_epoch='latest')
    steps_test = int(np.ceil(X_test.shape[0] / opt.BatchSize))

# Evaluate on each batch
# for h in [4, 8, 16, 32, 64, 128]:
n_batches = int(np.ceil(n_test / batch_size_test))
current = 0
nlls = []
for batch in range(n_batches):
    # model.num_inducing_closest = 32
    mean, cov = model.get_joint_prediction(X_test[current:(current+batch_size_test)])
    nlls.append(model.likelihood.negative_log_likelihood_multi(mean, cov, y_test[current:(current+batch_size_test)]))
    current += batch_size_test

# path_results = "results/{}/SVGP_{}/s{}/batchsize{}_H{}_M{}/".format(opt.dataset_name, opt.modelSVGP.lower().upper(),opt.dataset_nsplit, batch_size_test, h, opt.nip)
# if not os.path.exists(path_results):
#     os.makedirs(path_results)
np.save(path_results + f"nll.npy", np.mean(nlls))
