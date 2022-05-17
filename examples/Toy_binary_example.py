import numpy as np
import tensorflow as tf
import copy
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append("../")

#tf.config.run_functions_eagerly(True)
from svgp_nn_inducing.tf2.kernel.matern import MaternKernel
from svgp_nn_inducing.tf2.likelihoods import BernoulliLikelihood
from svgp_nn_inducing.tf2.sgp_vi import SVGP_Hensman, SVGP_NN, SVGP_SOLVE, SWSGP
from svgp_nn_inducing.tf2.utils import get_num_params

from sklearn.preprocessing import StandardScaler

np.random.seed(0)


######################## inputs ###############################
num_inducing_points = 4

num_inducing_pointsU = 2
num_inducing_pointsV = 2
batch_size = 50

num_inducing_closest = 2

num_inducing_points_nn = 2
hidden_size_n1 = 50
hidden_layer_n1 = 2


df_train = pd.read_csv("../data/banana/s0/banana_train_data.csv", header = None)
df_train_lables = pd.read_csv("../data/banana/s0/banana_train_labels.csv", header = None)
df_test = pd.read_csv("../data/banana/s0/banana_test_data.csv", header = None)
df_test_labels = pd.read_csv("../data/banana/s0/banana_test_labels.csv", header=None)
X_train, y_train = np.array(df_train), np.array(df_train_lables)
X_test , y_test = np.array(df_test), np.array(df_test_labels)

y_mean = 0
y_std = 1


if min(y_train) != -1 or max(y_train) != 1:
    raise NotImplementedError("For binary classification the outputs should be -1 or 1")


scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

inducing_points_init = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = num_inducing_points), :]
inducing_pointsU = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_pointsU), :]
inducing_pointsV = X_train[np.random.choice(np.arange(X_train.shape[0]), size=num_inducing_pointsV), :]

fig = plt.figure(figsize=(12 , 8))
for ii, label in enumerate(('A', 'B', 'C', 'D')):

    ax = fig.add_subplot(2, 2, ii + 1)

    if ii == 0:
        used_model = "VSGP"
    elif ii == 1:
        used_model = "IDSGP"
    elif ii == 2:
        used_model = "SWSGP"
    elif ii == 3:
        used_model = "SOLVE"


    ######################## model ###############################
    kernel = MaternKernel(length_scale = 0.0, noise_scale = 0.0, output_scale = 0.0, num_dims = X_train.shape[1])


    likelihood = BernoulliLikelihood(ngh = 50)


    if used_model == "VSGP":
        model_used = "VSGP"

        
        model = SVGP_Hensman(kernel, likelihood, inducing_points_init, X_train.shape[0], input_dim = X_train.shape[1],
                    y_mean = y_mean, y_std = y_std)
    elif used_model == "SOLVE":
        model_used = "SOLVE"

        
        model = SVGP_SOLVE(kernel, likelihood, inducing_pointsU, inducing_pointsV, X_train.shape[0], X_train.shape[1],
                           y_mean = y_mean, y_std = y_std)

    elif used_model == "SWSGP":
        model_used = "SWSGP"

        model = SWSGP(kernel, likelihood, inducing_points_init, num_inducing_closest, X_train.shape[0], input_dim = X_train.shape[1],
                    y_mean = y_mean, y_std = y_std)
    elif used_model == "IDSGP":
        model_used = "IDSGP"


        model = SVGP_NN(kernel, likelihood, num_inducing_points_nn, X_train.shape[0], input_dim = X_train.shape[1],
                    y_mean = y_mean, y_std = y_std, n_hidden1 = hidden_size_n1, n_layers1 = hidden_layer_n1,
                        dropout_rate=0)

    np.random.seed(0)


    epochs = 200

    model(X_train[0:batch_size])
    num_pars = get_num_params(model.trainable_variables)
    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  run_eagerly=False)
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs)


    ######################## Prediciton ###############################

    X = X_test
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    ghat, gvar = model.predict(grid)
    ghat[ghat <= 0.5] = -1
    ghat[ghat > 0.5] = 1
    # reshape the predictions back into a grid
    zz = ghat.reshape(xx.shape)



    ######################## Plot the results ###############################

    if model_used == "VSGP":
        inducing_points = model.inducing_points.numpy()
        # plot the grid of x, y and z values as a surface
        level_intervals = [0, 10, 20, 30, 40, 50]
        ax.contour(xx, yy, zz, colors='black', linewidths=1.5, levels=level_intervals)

        ax.scatter(X_test[:, 0], X_test[:, 1], marker='x', label='Test data', c=y_test)


        ax.scatter(inducing_points[:, 0], inducing_points[:, 1], marker='o', s=50, label='Inducing points',
                    c='red')
        
        file_name = 'TITSIAS_experiment_NIP_%d_batchsize_%d' % \
                    (num_inducing_points, batch_size)
        ax.legend()

    elif model_used == "SWSGP":
        inducing_points = model.inducing_points.numpy()
    
        # plot the grid of x, y and z values as a surface
        level_intervals = [0, 10, 20, 30, 40, 50]
        ax.contour(xx, yy, zz, colors='black', linewidths=1.5, levels=level_intervals)

        ax.scatter(X_test[:, 0], X_test[:, 1], marker='x', label='Test data', c=y_test)

        ax.scatter(inducing_points[:, 0], inducing_points[:, 1], marker='o', s=50, label='Inducing points',
                    c='red')

        file_name = 'SWSGP_experiment_Nclose_%d_NIP_%d_batchsize_%d' % \
                    (num_inducing_closest,num_inducing_points, batch_size)


    elif model_used == "SOLVE":
        inducing_points_u = model.inducing_points_u
        inducing_points_v = model.inducing_points_v

        # plot the grid of x, y and z values as a surface
        level_intervals = [0, 10, 20, 30, 40, 50]
        ax.contour(xx, yy, zz, colors='black', linewidths=1.5, levels=level_intervals)

        ax.scatter(X_test[:, 0], X_test[:, 1], marker='x', label='Test data', c=y_test)


        ax.scatter(inducing_points_u[:, 0], inducing_points_u[:, 1], marker='o', s=50, label='Inducing points u',
                    c='red')
        ax.scatter(inducing_points_v[:, 0], inducing_points_v[:, 1], marker='o', s=50, label='Inducing points v',
                    c='green')

        file_name = 'SOLVE_experiment_NIP_%d_NIPv_%d_batchsize_%d' % \
                    (num_inducing_pointsU, num_inducing_pointsV, batch_size)



    elif model_used == "IDSGP":
        d = model.net(X_test)
        output_net1 = d[:,:model.num_inducing * model.input_dim]
        inducing_points_t = tf.reshape(output_net1, tf.stack([tf.shape(input=output_net1)[0], model.num_inducing,
                                   tf.cast(tf.shape(input=output_net1)[1] / model.num_inducing, dtype = tf.int32)]))


        for i in range(1):
            inducing_points = inducing_points_t[i,:, :].numpy()

            # plot the grid of x, y and z values as a surface
            level_intervals = [0, 10, 20, 30, 40, 50]
            ax.contour(xx, yy, zz, colors='black', linewidths=1.5, levels=level_intervals)

            ax.scatter(X_test[:, 0], X_test[:, 1], marker='x', label='Test data', c=y_test)


            ax.scatter(inducing_points[:, 0], inducing_points[:, 1], marker='o', s=50, label='Inducing points for point X',
                        c='red')
            ax.scatter(X_test[i, 0], X_test[i, 1], marker='o', s=90, label='point X', c='green')

            file_name = 'NN_experiment_hs1_%d_hl1_%d_NIP_%d_batchsize_%d' % \
                        (hidden_size_n1, hidden_layer_n1, num_inducing_points, batch_size)
    
    ax.set_title(used_model, fontsize=16, fontweight='bold')
    ax.legend( loc="upper left", prop={'size': 11})

fig.tight_layout()
file_name = 'Banana_SWSGP_Nc_%d_SOLVE_Nu_%d_Nv_%d_NN_%d_%d_%d_Titsias_NIP_%d_batchsize_%d' % \
            (num_inducing_closest, num_inducing_pointsU, num_inducing_pointsV,
             hidden_size_n1, hidden_layer_n1, num_inducing_points_nn, num_inducing_points, batch_size)

plt.savefig(file_name + '_{}.pdf'.format(i), dpi=300)

