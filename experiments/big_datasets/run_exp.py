import numpy as np
import tensorflow as tf
import copy
import time
import pandas as pd
import sys
import os

sys.path.append("../../")

from svgp_nn_inducing.tf2.kernel.matern import MaternKernel
from svgp_nn_inducing.tf2.kernel.rbf_ard import RBF_ARD
from svgp_nn_inducing.tf2.likelihoods import GaussianLikelihood, BernoulliLikelihood, BernoulliLikelihood_sigmoid
from svgp_nn_inducing.tf2.sgp_vi import SVGP_Titsias, SVGP_NN, SVGP_SOLVE, SWSGP
from svgp_nn_inducing.tf2.utils import get_num_params, save_model, load_model
from svgp_nn_inducing.tf2.callbacks import NBatchCSVLogger
from svgp_nn_inducing.tf2.dataset_generator import BigDatasetLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.config.run_functions_eagerly(True)

np.random.seed(0)

#### Exapmle ####
#  python dataset num_dataset method (reg or class) (MinMax or MeanStd) GPU_Num

######################## inputs ###############################
dataset_name = sys.argv[-6]
i = int(sys.argv[-5])
model_used = sys.argv[-4]
regression_classification = sys.argv[-3]
standardize_input = sys.argv[-2]
GPU_N = sys.argv[-1]
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_N
if regression_classification == 'reg':
    likelihood = GaussianLikelihood()
elif regression_classification == 'class':
    #likelihood = BernoulliLikelihood(ngh = 50)
    likelihood = BernoulliLikelihood_sigmoid(ngh = 50)

if model_used == "nn":
    path_results = "results/{}/SVGP_NN/".format(dataset_name)
elif model_used == 'titsias':
    path_results = "results/{}/SVGP_Titsias/".format(dataset_name)
elif model_used.lower() == 'solve':
    path_results = "results/{}/SVGP_SOLVE/".format(dataset_name)
elif model_used.lower() == 'swsgp':
    path_results = "results/{}/SVGP_SWSGP/".format(dataset_name)
    
    
if not os.path.exists(path_results):
    os.makedirs(path_results)

batch_size = 100

if regression_classification == 'reg':
    standardize_output = True
else:
    standardize_output = False

# Load data
dataset_loader = BigDatasetLoader("../../data/{}".format(dataset_name), batch_size, shuffle=False, standardize_input = True, standardize_output=standardize_output)
data_train = dataset_loader.get_train()
data_test = dataset_loader.get_test()

y_mean = dataset_loader.y_mean
y_std = dataset_loader.y_std

######################## model ###############################
# We estimate the log length scales
log_l = dataset_loader.estimate_lengthscale()
# kernel = RBF_ARD(log_lengthscales = log_l, log_sigma0 = 0.0, log_sigma = 0.0)
kernel = MaternKernel(length_scale = 0.0,
                    noise_scale = 0.0, output_scale = 0.0, num_dims = dataset_loader.n_attributes)

epoch = 10000
if model_used == 'titsias':
    hidden_size_n1 = None
    hidden_size_n2 = None
    hidden_layer_n1 = None
    hidden_layer_n2 = None
    num_inducing_points = 1024
    inducing_points = dataset_loader.sample(num_inducing_points)
    inducing_points_prior = copy.deepcopy(inducing_points)
    model = SVGP_Titsias(kernel, likelihood, inducing_points, dataset_loader.n_train, dataset_loader.n_attributes, y_mean, y_std, path_results= path_results)
    file_name = 'experiment_NIP_%d_batchsize_%d' % (num_inducing_points, batch_size)
    print('NIP {}, BS {}'.format(num_inducing_points, batch_size))
elif model_used == 'solve':
    num_inducing_points_u = 1024
    num_inducing_points_v = 1024
    inducing_points = dataset_loader.sample(num_inducing_points_u+num_inducing_points_v)
    inducing_points_prior = copy.deepcopy(inducing_points)
    inducing_points_u = inducing_points[0:num_inducing_points_u]
    inducing_points_v = inducing_points[num_inducing_points_u:]
    model = SVGP_SOLVE(kernel, likelihood, inducing_points_u, inducing_points_v,  dataset_loader.n_train, dataset_loader.n_attributes, y_mean, y_std, path_results= path_results)
    file_name = 'experiment_NIPu_%d_v_%d_batchsize_%d' % (num_inducing_points_u,
                                                             num_inducing_points_v, batch_size)
    print('NIP_h {}, NIP_v {}, BS {}'.format(num_inducing_points_u, num_inducing_points_v, batch_size))

elif model_used == 'swsgp':
    num_inducing_points = 1024
    num_inducing_closest = 50
    inducing_points = dataset_loader.sample(num_inducing_points)
    model = SWSGP(kernel, likelihood, inducing_points, num_inducing_closest, dataset_loader.n_train, dataset_loader.n_attributes,
                 y_mean, y_std, n_hidden1=15, n_layers1=2, n_hidden2=15, n_layers2=2, path_results='', seed=0)


    file_name = 'experiment_NIP_total_%d_closest_%d_batchsize_%d' % (num_inducing_points,
                                                             num_inducing_closest, batch_size)
    print('NIP_total {}, NIP_closest {}, BS {}'.format(num_inducing_points, num_inducing_closest, batch_size))

elif model_used == 'nn':
    hidden_size_n1 = 25
    hidden_layer_n1 = 2
    num_inducing_points = 50
    model = SVGP_NN(kernel, likelihood, num_inducing_points, dataset_loader.n_train, dataset_loader.n_attributes,
                   y_mean, y_std, hidden_size_n1, hidden_layer_n1, path_results, dropout_rate=0.0)
    print('H1 {}, L1 {}, NIP {}, BS {}'.format(hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size))
    file_name = 'experiment_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                (hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)


# Initialize model
model(dataset_loader.sample(batch_size))

num_pars = get_num_params(model.trainable_variables)
model.summary()

np.random.seed(0)

fitting = True
optimizer = tf.optimizers.Adam(learning_rate=1e-4)

if fitting:
    start = time.time()
    model.compile(optimizer=optimizer, run_eagerly=False)

    steps_test = int(np.ceil(dataset_loader.n_test / batch_size))
    steps_train = int(np.ceil(dataset_loader.n_train / batch_size))

    logger = NBatchCSVLogger(data_test, batch_size, steps_test, path_results + file_name + ".txt")
    
    history = model.fit(data_train, 
                        #batch_size=batch_size, 
                        epochs=epoch, 
                        steps_per_epoch=steps_train,
                        callbacks=[logger]
                        )
    save_model(model, optimizer, path_results)

    end = time.time()

else:
    load_model(model, optimizer, model.path_results, which_epoch='latest')
    model.compile(optimizer=optimizer, run_eagerly=True)




