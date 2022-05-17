import numpy as np
import tensorflow as tf
import copy
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("../")

# tf.config.run_functions_eagerly(True)
from svgp_nn_inducing.tf2.kernel.matern import MaternKernel
from svgp_nn_inducing.tf2.kernel.matern import MaternKernel
from svgp_nn_inducing.tf2.likelihoods import GaussianLikelihood
from svgp_nn_inducing.tf2.sgp_vi import SVGP_Hensman, SVGP_NN, SVGP_SOLVE, SWSGP
from svgp_nn_inducing.tf2.utils import get_num_params
#from svgp_nn_inducing.tf2.callbacks import CustomCallback

np.random.seed(0)


######################## inputs ###############################
num_inducing_points = 4

num_inducing_pointsU = 2
num_inducing_pointsV = 2
batch_size = 50
from sklearn.preprocessing import StandardScaler
num_inducing_closest = 2

#num_inducing_points_nn = 2
hidden_size_n1 = 50
hidden_layer_n1 = 2
#num_inducing_points_nns = [2, 4, 6, 8, 10, 12]+np.arange(15, 105, 10).tolist()+[100]
num_inducing_points_nns = [2, 4,8, 16]
#num_inducing_points_nns = [ 32,64,128]
fig = plt.figure(figsize=(24 , 12))
for ii in range(len(num_inducing_points_nns)):
    ax = fig.add_subplot(2, 2, ii + 1)
    #ax.text(-0.1, 1.15, label, transform=ax.transAxes,
     # fontsize=16, fontweight='bold', va='top', ha='right')

    used_model = "SOLVE"

    num_inducing_points_nn = num_inducing_points_nns[ii]
    num_inducing_points = num_inducing_points_nns[ii]
    num_inducing_pointsU = num_inducing_points_nns[ii]
    num_inducing_pointsV = num_inducing_points_nns[ii]
    if used_model == "SWSGP":
        num_inducing_points = 128
        num_inducing_closest = num_inducing_points_nns[ii]
    data_train = pd.read_csv("../data/1D_toy_data/Simple_train.csv")
    data_test = pd.read_csv("../data/1D_toy_data/Simple_test.csv")

    confid_region = pd.read_csv("../data/1D_toy_data/Confidence region_test.csv")

    X_test = data_test['X'].values.reshape(-1 , 1)
    y_test = data_test['Y'].values
    X_train = data_train['X'].values.reshape(-1 , 1)
    y_train = data_train['Y'].values
    #x_mean, x_std = X_train.mean(), X_train.std()
    #X_train = (X_train - x_mean)/x_std
    #X_test = (X_test - x_mean) / x_std

    y_mean = 0
    y_std = 1

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #y_mean = np.mean(y_train)
    #y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std




    ######################## model ###############################
    # We estimate the log length scales
    X_sample = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = 1000), :  ]
    dist2 = np.sum(X_sample**2, 1, keepdims = True) - 2.0 * np.dot(X_sample, X_sample.T) + np.sum(X_sample**2, 1, keepdims = True).T
    log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(1000, 1) ]))
    #ernel = MaternKernel(length_scale=log_l, noise_scale = 0.0, output_scale =-0.3667252797922339)
    kernel = MaternKernel(length_scale=0.0,
                          noise_scale=0.0, output_scale=0.0, num_dims=X_train.shape[1])
    #kernel = RBF_ARD(log_l * np.ones(X_train.shape[ 1 ]), -1.0, 1.0)

    likelihood = GaussianLikelihood()

    inducing_points = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = num_inducing_points), :]

    if used_model == "VSGP":
        used_model = "VSGP"
        model = SVGP_Hensman(kernel, likelihood, inducing_points, X_train.shape[0], input_dim = X_train.shape[1],
                    y_mean = y_mean, y_std = y_std)
    elif used_model == "IDSGP":

        used_model = "IDSGP"

        model = SVGP_NN(kernel, likelihood, num_inducing_points_nn, X_train.shape[0], input_dim = X_train.shape[1],
                    y_mean = y_mean, y_std = y_std, n_hidden1 = hidden_size_n1, n_layers1 = hidden_layer_n1,
                        dropout_rate=0.0)
        d = model.net(X_train)
        output_net1 = d[:,:model.num_inducing * model.input_dim]
        inducing_points_prior = output_net1[100,:]

    elif used_model == "SOLVE":
        used_model = "SOLVE"

        inducing_pointsU = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_pointsU), :]
        inducing_pointsV = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_pointsV), :]
        model = SVGP_SOLVE(kernel, likelihood, inducing_pointsU, inducing_pointsV, X_train.shape[0], X_train.shape[1],
                           y_mean = y_mean, y_std = y_std)
    elif used_model == "SWSGP":
        used_model = "SWSGP"

        model = SWSGP(kernel, likelihood, inducing_points, num_inducing_closest, X_train.shape[ 0 ], X_train.shape[1],
                     y_mean, y_std, seed=0)



    np.random.seed(0)

    model(X_train[0:50])
    num_pars = get_num_params(model.trainable_variables)
    model.summary()

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), run_eagerly=True)
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, shuffle = True,
                        epochs=500)


    ######################## Prediciton ###############################
    if used_model != "IDSGP":
        inducing_points_prior = inducing_points
    y_pred_test, y_var_test = model(X_test)

    if used_model == "IDSGP":
        d = model.net(X_train)
        X_train[100]
        output_net1 = d[:,:model.num_inducing * model.input_dim]
        inducing_points = output_net1[100,:]
        if isinstance(inducing_points, tf.Variable):
            inducing_points = inducing_points.numpy()
        file_name = 'Toy_NN_experiment_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                    (hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)
    elif used_model == "SOLVE":
        inducing_pointsU = model.inducing_points_u
        inducing_pointsV = model.inducing_points_v
        if isinstance(inducing_pointsU, tf.Variable):
            inducing_pointsU = inducing_pointsU.numpy()
            inducing_pointsV = inducing_pointsV.numpy()
        file_name = 'Toy_SOLVE_experiment_Nu_%d_Nv_%d_batchsize_%d' % \
                    (num_inducing_pointsU, num_inducing_pointsV, batch_size)
    elif used_model == "VSGP":
        inducing_points = model.inducing_points

        if isinstance(inducing_points, tf.Variable):
            inducing_points = inducing_points.numpy()
        file_name = 'Toy_TITSIAS_experiment_NIP_%d_batchsize_%d' % \
                    (num_inducing_points, batch_size)
    elif used_model == "SWSGP":
        inducing_points = model.inducing_points

        if isinstance(inducing_points, tf.Variable):
            inducing_points = inducing_points.numpy()
        file_name = 'Toy_SWSGP_experiment_Nc_%d_NIP_%d_batchsize_%d' % \
                    (num_inducing_closest, num_inducing_points, batch_size)
    ######################## back transform the data ###############################
    # back transform the data
    pred_test = y_pred_test * y_std + y_mean
    y_var_test = y_var_test * y_std
    x_std = 1
    x_mean = 0
    X_train = X_train*x_std + x_mean
    X_test = X_test*x_std + x_mean

    if used_model== "SOLVE":
        inducing_pointsU = inducing_pointsU*x_std + x_mean
        inducing_pointsV = inducing_pointsV * x_std + x_mean
    else:
        inducing_points = inducing_points * x_std + x_mean
    inducing_points_prior = inducing_points_prior * x_std + x_mean

    ######################## Plot the results ###############################
    #plt.figure(figsize=(10 , 6))
    #plt.subplot(2,2,4)
    ax.scatter(X_train, y_train, marker = 'x' ,label = 'Training data')

    ax.plot(X_test, y_pred_test, 'b' , label = 'mean prediction')
    ax.plot(X_test, y_test, linestyle = '--', color = 'b' , label = 'mean GP prediction')
    lower, upper = y_pred_test - 2*np.sqrt(y_var_test), y_pred_test + 2*np.sqrt(y_var_test)


    ax.plot(X_test, lower, '--' ,  color='red' , label = 'Standard deviation')
    ax.plot(X_test, upper, '--' , color='red')


    lowerO, upperO = confid_region['Lower'], confid_region['Upper']


    ax.plot(X_test, lowerO, '--' ,  color='brown' , label = 'Standard deviation GP')
    ax.plot(X_test, upperO, '--' , color='brown')



    min_lower_std = np.min(lower) - 0.1
    max_upper_std = np.max(upper) + 0.1
    ax.scatter(inducing_points_prior, max_upper_std*np.ones_like(inducing_points_prior),
                        s=14, marker="x", label="Initial inducing points")

    if used_model == "IDSGP":
        ax.scatter(inducing_points, min_lower_std*np.ones_like(inducing_points),
                        s=14, marker="o", label="Inducing points for X", c='magenta')
        ax.scatter(X_train[100], y_train[100], marker='*', s=80, label='point X', c='red')

    elif used_model == "SOLVE":
        ax.scatter(inducing_pointsU, min_lower_std*np.ones_like(inducing_pointsU),
                        s=14, marker="o", label="Inducing points U", c='magenta')
        ax.scatter(inducing_pointsV, min_lower_std*np.ones_like(inducing_pointsV),
                        s=14, marker="o", label="Inducing points V", c='green')
    else:
        ax.scatter(inducing_points, min_lower_std*np.ones_like(inducing_points),
                        s=14, marker="o", label="Inducing points", c = 'magenta')
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    font_dict = {'fontsize': 12,
     'fontweight': 'bold'}
    if used_model == "SWSGP":
        ax.set_title(used_model + ' (number of closest inducing points = {}, total number of inducing points = {}) '.format(num_inducing_points_nn, num_inducing_points), fontdict=font_dict)
    elif used_model == "SOLVE":
        ax.set_title(used_model + ' (inducing points U = {}, inducing points V = {}) '.format(num_inducing_pointsU, num_inducing_pointsV), fontdict=font_dict)
    elif used_model == "IDSGP":
        ax.set_title(used_model + ' (number of inducing points for each point = {}) '.format(num_inducing_points_nn), fontdict=font_dict)
    else:
        ax.set_title(used_model + ' (number of inducing points = {}) '.format(num_inducing_points_nn), fontdict=font_dict)
    ax.legend( loc="upper right", prop={'size': 8,'weight': 'bold'})

    file_name = 'Toy_%s_Nc_%d_SOLVE_Nu_%d_Nv_%d_NN_%d_%d_Titsias_NIP_%d_batchsize_%d' % \
                (used_model, num_inducing_closest, num_inducing_pointsU, num_inducing_pointsV,
                 hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)

    plt.savefig(file_name + '.pdf', format='pdf')
    plt.close()

    #plt.show()
#plt.title("1D toy Problem ")



