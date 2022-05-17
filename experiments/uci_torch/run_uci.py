import numpy as np
import torch
import copy
import time
import pandas as pd
import itertools
import sys
import os
sys.path.append("../../")
from svgp_nn_inducing.torch.models.sgp_vi import SVGP_Titsias, SVGP_NN, SVGP_SOLVE, SVGP_SWSGP
from svgp_nn_inducing.torch.kernels.Matern_kernel import MaternKernel
from svgp_nn_inducing.torch.options import options
from svgp_nn_inducing.torch.utils.utils import *
from svgp_nn_inducing.torch.utils.train import train
from svgp_nn_inducing.torch.utils.data_loader import CDataSet
from svgp_nn_inducing.torch.models.likelihoods import GaussianLikelihood, BernoulliLikelihood, BernoulliLikelihood_sigmoid, \
    RobustLikelihood
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
#-m torch.multiprocessing.spawn

np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

opt = options().parse()
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and opt.nGPU>=0:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")

print('devcie :{}'.format(device.type))

initial_seed = True
if initial_seed:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)




def main():



    ######################## inputs ###############################
    
    dataset_name = opt.dataset_name

    if opt.Ptype.lower() == 'reg':
        likelihood = GaussianLikelihood(device = device)
    elif opt.Ptype.lower() == 'class':
        if opt.ll.lower() == 'bern':
            likelihood = BernoulliLikelihood(ngh=50 , device=device)
        elif opt.ll.lower() == 'bern_sig':
            likelihood = BernoulliLikelihood_sigmoid(ngh=50,device = device)
        elif opt.ll.lower() == 'robust':
            likelihood = RobustLikelihood(device = device)
        else:
            raise Exception('Please define a propser likelihood for classification.')

    path_results = "results/{}/SVGP_{}/s{}/".format(opt.dataset_name, opt.modelSVGP.lower().upper(), opt.dataset_nsplit)
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    df_train = pd.read_csv(
        "../../data/{}/s{}/{}_train_data.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name),
        header=None)
    df_train_lables = pd.read_csv(
        "../../data/{}/s{}/{}_train_labels.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name),
        header=None)
    df_test = pd.read_csv(
        "../../data/{}/s{}/{}_test_data.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name),
        header=None)
    df_test_labels = pd.read_csv(
        "../../data/{}/s{}/{}_test_labels.csv".format(opt.dataset_name, opt.dataset_nsplit, opt.dataset_name),
        header=None)
    X_train, y_train = np.array(df_train), np.array(df_train_lables).squeeze()
    X_test, y_test = np.array(df_test), np.array(df_test_labels).squeeze()


    if np.ndim(y_train)== 1:
        y_train = np.expand_dims(y_train, -1)

    if np.ndim(y_test)== 1:
        y_test = np.expand_dims(y_test, -1)
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
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif opt.scaling.lower() == 'maxabs':
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif opt.scaling.lower() == 'robust':
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    ######################## model ###############################
    if opt.kernel == 'matern':
        kernel = MaternKernel(length_scale=0.0,
                              noise_scale=0.0, output_scale=0.0, num_dims=X_train.shape[1], device=device)
    elif opt.kernel == 'rbf':
        kernel = RBF_ARD(log_lengthscales=[0.0] * X_train.shape[1],
                         log_sigma0=0.0, log_sigma=0.0, device=device)
    else:
        raise Exception('not implemented.')
    if opt.modelSVGP.lower() == 'titsias':

        num_inducing_points = opt.nip
        batch_size = opt.BatchSize
        inducing_points = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_points), :].astype(
            np.float64)
        inducing_points_prior = copy.deepcopy(inducing_points)
        model = SVGP_Titsias(kernel, likelihood, inducing_points, X_train.shape[0], X_train.shape[1], y_mean, y_std,
                             path_results=path_results, device= device)
        file_name = 'experiment_ll_%s_NIP_%d_batchsize_%d' % (opt.ll.lower(), num_inducing_points, batch_size)
        print('NIP {}, BS {}'.format(num_inducing_points, batch_size))
    elif opt.modelSVGP.lower() == 'solve':

        num_inducing_points_u = opt.nip
        num_inducing_points_v = opt.nip
        batch_size = opt.BatchSize
        inducing_points = X_train[np.random.choice(np.arange(X_train.shape[0]),
                                                   size=num_inducing_points_u + num_inducing_points_v), :].astype(
            np.float64)
        inducing_points_prior = copy.deepcopy(inducing_points)
        inducing_points_u = inducing_points[0:num_inducing_points_u]
        inducing_points_v = inducing_points[num_inducing_points_u:]
        model = SVGP_SOLVE(kernel, likelihood, inducing_points_u, inducing_points_v, X_train.shape[0], X_train.shape[1],
                           y_mean, y_std, path_results=path_results, device=device)
        file_name = 'experiment_ll_%s_NIPu_%d_v_%d_batchsize_%d' % (opt.ll.lower(), num_inducing_points_u,
                                                                    num_inducing_points_v, batch_size)
        print('NIP_h {}, NIP_v, BS {}'.format(num_inducing_points_u, num_inducing_points_v, batch_size))

    elif opt.modelSVGP.lower() == 'swsgp':

        num_inducing_points = opt.nip
        num_inducing_closest = opt.ncip
        batch_size = opt.BatchSize
        inducing_points = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_points), :].astype(
            np.float64)
        model = SVGP_SWSGP(kernel, likelihood, inducing_points, num_inducing_closest, X_train.shape[0], X_train.shape[1],
                      y_mean, y_std, path_results='', seed=0, device=device)

        file_name = 'experiment_ll_%s_NIP_total_%d_closest_%d_batchsize_%d' % (opt.ll.lower(), num_inducing_points,
                                                                               num_inducing_closest, batch_size)
        print('NIP_total {}, NIP_closet, BS {}'.format(num_inducing_points, num_inducing_closest, batch_size))

    elif opt.modelSVGP.lower() == 'nn':
        hidden_size_n1 = opt.nhn1
        hidden_layer_n1 = opt.nhl1

        num_inducing_points = opt.nip

        batch_size = opt.BatchSize
        model = SVGP_NN(kernel, likelihood, num_inducing_points, X_train.shape[0], X_train.shape[1],
                        y_mean, y_std, hidden_size_n1, hidden_layer_n1, path_results,
                        dropout_rate=opt.rdropout, device= device)
        print('H1 {}, L1 {}, NIP {}, BS {}'.format(hidden_size_n1, hidden_layer_n1,
                                                                  num_inducing_points, batch_size))
        file_name = 'experiment_ll_%s_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                    (opt.ll.lower(), hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)

    params_train = {'batch_size': opt.BatchSize,
                    'shuffle': False,
                    #'pin_memory': True,
                    # 'collate_fn': collate_wrapper,
                    }

    training_set = CDataSet(X_train, y_train, device= device)
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)
    params_test = {'batch_size': opt.BatchSize,
                   'shuffle': False,
                   #'pin_memory': True,
                   # 'collate_fn': collate_wrapper
                   }
    # test_set = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype= settings.torch_float_type),
    #                                             torch.tensor(y_test, dtype= settings.torch_float_type))
    test_set = CDataSet(X_test, y_test, device= device)
    test_generator = torch.utils.data.DataLoader(test_set, **params_test)

    # Initialize model
    model(torch.from_numpy(X_train[0:opt.BatchSize]).to(device).to(settings.torch_float_type))


    np.random.seed(0)

    fitting = True
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, betas= (opt.b1, opt.b2), weight_decay=0.00)


    if fitting:
        start = time.time()
        path_file_name = path_results + file_name + ".txt"
        #model.compile(optimizer=optimizer)
        #logger = CustomCallback(X_train, y_train, X_test, y_test, opt.BatchSize, path_results + file_name + ".txt")  # ,
        train(model, training_generator, test_generator, optimizer, path_file_name, batch_size, epochs=opt.nEpoch, silent=False, rank=0)
        # predict_test=False)

        save_model(model, optimizer, device, path_results)

        end = time.time()

    else:
        load_model(model, optimizer, model.path_results, which_epoch='latest')
        model.compile(optimizer=optimizer, run_eagerly=True)


if __name__ == '__main__':
    main()
