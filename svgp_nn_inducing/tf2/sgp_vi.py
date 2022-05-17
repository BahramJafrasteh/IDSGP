
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import svgp_nn_inducing.tf2.settings as settings
from svgp_nn_inducing.tf2.utils import get_num_params
import abc

class SVGP(tf.keras.Model):
    '''
    Abstract class for a SVGP model
    '''

    def __init__(self, model_name, likelihood, y_mean, y_std, seed, dtype = tf.float64):
        super(SVGP, self).__init__(model_name, dtype=dtype)

        # Random seed
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Likelihood
        self.likelihood = likelihood

        # Save y_mean and y_std to denormalize data when predicting
        self.y_mean = y_mean # mean target
        self.y_std = y_std # standard deviation target

        # Metric trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="nelbo")

    @abc.abstractmethod
    def nelbo(self, inputs, outputs):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def call(self, inputs, outputs):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def _init_parameters(self):
        """
        Initializes the trainable parameters of the model
        """
        raise NotImplementedError("Subclass should implement this.")

    @tf.function
    def train_step(self, data):
        x, y = data
        
        if len(y.shape) == 2:
            y = y[:,0]
        if eval("tf." + self.dtype) != x.dtype:
            x = tf.cast(x, self.dtype)
        if eval("tf." + self.dtype) != y.dtype:
            y = tf.cast(y, self.dtype)

        with tf.GradientTape() as tape:
            mean_pred, var_pred = self(x, training=True) # Forward pass

            # Compute our own loss
            loss = self.nelbo(x, y)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.likelihood.update_metrics(y * self.y_std + self.y_mean, mean_pred, var_pred)
        
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        if len(y.shape) == 2:
            y = y[:,0]
        if eval("tf." + self.dtype) != x.dtype:
            x = tf.cast(x, self.dtype)
        if eval("tf." + self.dtype) != y.dtype:
            y = tf.cast(y, self.dtype)

        # Compute predictions
        mean_pred, var_pred = self(x, training=False) # Forward pass

        # Compute the loss
        loss = self.nelbo(x, y)

        # Update the metrics
        self.loss_tracker.update_state(loss)
        self.likelihood.update_metrics(y * self.y_std + self.y_mean, mean_pred, var_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [self.loss_tracker]
        for metric in self.likelihood.metrics:
            metrics.append(metric)
        
        return metrics

class FullGP(SVGP):
    """
    Full GPR
    """

    def __init__(self, kernel, likelihood, total_training_data, inputs, outputs, y_mean, y_std, path_results='', seed=0):
        super(FullGP, self).__init__('FullGP', likelihood, y_mean, y_std, seed=seed, dtype = settings.tf_float_type)

        self.kernel = kernel

        # number of training points
        self.total_training_data = tf.constant([ 1.0 * total_training_data ], dtype = settings.tf_float_type)

        self.inputs = inputs
        self.outputs = outputs
        
        self.likelihood.build(total_training_data)
        self.initialized = True
    
    @tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the marginal likelihood
        """
        inputs_train = self.inputs
        outputs = tf.reshape(self.outputs, (-1, 1))

        Kf = self.kernel(inputs_train)
        Kf = Kf + tf.eye(Kf.shape[0], dtype = self.dtype) * tf.exp(self.likelihood.log_var_noise)
        chol_Kf = tf.linalg.cholesky(Kf)

        Ks = self.kernel(inputs)
        Kfs = self.kernel(inputs_train, inputs)

        KfinvKfs = tf.linalg.cholesky_solve(chol_Kf, Kfs)
        Kfinvy = tf.linalg.cholesky_solve(chol_Kf, outputs)

        KsfKfinvKfs = tf.matmul(Kfs, KfinvKfs, transpose_a = True)
        KsfKfinvy = tf.matmul(Kfs, Kfinvy, transpose_a = True)

        return (Ks, chol_Kf, Kfinvy, KsfKfinvy, KsfKfinvKfs)


    @tf.function
    def nelbo(self, inputs, outputs):
        """
        @return: Marginal likelihood of GPR
        """
        
        # We use self.inputs and self.outputs so it
        # is always computed on the training set
        matrix_params = self.update_params(self.inputs)

        outputs = tf.reshape(self.outputs, (-1, 1))

        _, chol_Kf, Kfinvy, _, _ = matrix_params

        mll = -0.5 * tf.squeeze(tf.matmul(outputs, Kfinvy, transpose_a=True))
        mll += - tf.reduce_sum(input_tensor=tf.math.log( tf.linalg.diag_part(chol_Kf)), axis=[-1])
        mll += -0.5 * self.total_training_data * tf.math.log(2 * tf.constant(np.pi, dtype = self.dtype))

        return - mll

    @tf.function
    def call(self, inputs):
        """
        @return: Predictive mean and variance of the GP model
        """

        matrix_params = self.update_params(inputs)
        
        mean_pred, var_pred = self._build_predictive_gaussian_mean_vars(inputs, matrix_params)

        # y_train was standardize so we need to back transform the predictions
        mean_pred = mean_pred * self.y_std + self.y_mean
        var_pred *= self.y_std**2

        return mean_pred, var_pred

    
    @tf.function
    def _build_predictive_gaussian_mean_vars(self, inputs, matrix_params):
        '''
        Computes the mean and variance of E_q p(y|f) 
        '''
        
        Ks, _, _, KsfKfinvy, KsfKfinvKfs = matrix_params

        # We get the marginals of the values of the process
        
        means = tf.squeeze(KsfKfinvy)
         
        marginal_variances = tf.linalg.diag_part(Ks)
        
        variances = - tf.linalg.diag_part(KsfKfinvKfs) + marginal_variances

        return means, variances


    def get_joint_prediction(self, inputs):
        
        matrix_params = self.update_params(inputs)

        Ks, _, _, KsfKfinvy, KsfKfinvKfs = matrix_params

        means = tf.squeeze(KsfKfinvy)

        covariances = - KsfKfinvKfs + Ks

        return means, covariances


class SVGP_Titsias(SVGP):
    '''
    Variational Inference regression with sparse gaussian processes
    '''
    
    def __init__(self, kernel, likelihood, inducing_points, total_training_data, input_dim, inputs, outputs, y_mean, y_std, path_results='', seed=0):

        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood used for the model
        @param inducing_points: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        '''
        super(SVGP_Titsias, self).__init__('SVGP_Titsias', likelihood, y_mean, y_std, seed=seed, dtype = settings.tf_float_type)

        self.kernel = kernel

        # number of training points
        self.total_training_data = tf.constant([ 1.0 * total_training_data ], dtype = settings.tf_float_type)

        # Repeat the inducing inputs for all latent processes if we haven't been given individually
        # specified inputs per process.
        self.num_inducing = inducing_points.shape[0] # number of inducing points
        self.input_dim = input_dim # input dimension
        
        # Inducing points initialization
        self.inducing_points = inducing_points

        self.inputs = inputs
        self.outputs = outputs
        
        self.initialized = False

    def _init_parameters(self):
        """
        Initializes the trainable parameters of the model
        """

        # update Kernels and matrix computation
        self.inducing_points = tf.Variable(self.inducing_points, dtype = settings.tf_float_type, name="inducing_points")
     
        self.initialized = True
        
    @tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """
        inputs_train = self.inputs
        outputs = tf.reshape(self.outputs, (-1, 1))

        Knm_train = self.kernel(inputs_train, self.inducing_points)  # K_{nm}
        Knm = self.kernel(inputs, self.inducing_points)  # K_{nm}
        Kmm = self.kernel(self.inducing_points)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}

        
        chol_Sigmap = tf.linalg.cholesky(Kmm + tf.matmul(Knm_train, Knm_train, transpose_a = True) / tf.exp(self.likelihood.log_var_noise))
        SigmaInvKmm = tf.linalg.cholesky_solve(chol_Sigmap, Kmm)
        KmmSigmaInvKmm = tf.matmul(Kmm, SigmaInvKmm)
        KnmSigmaInvKmm = tf.matmul(Knm_train, SigmaInvKmm)
        m = tf.matmul(KnmSigmaInvKmm, outputs, transpose_a = True) / tf.exp(self.likelihood.log_var_noise)
        L = tf.linalg.cholesky(KmmSigmaInvKmm)
  
        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, m)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(a=Knm))
        LtKmmInvKmn = tf.matmul(tf.transpose(a=L), KmmInvKmn)

        return (Knm, chol_Kmm, m, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)


    @tf.function
    def nelbo(self, inputs, outputs):
        """
        @return: NELBO for the current model
        """
        
        if not self.initialized:
            self._init_parameters()

        matrix_params = self.update_params(inputs)

        _, chol_Kmm, m, L, _, _, _ = matrix_params

        KL_sums = self._build_KL_objective(chol_Kmm, m, L)
        expected_log_terms = self._build_expected_log_term(inputs, outputs, matrix_params)
        
        n_train = tf.shape(input=inputs)[0]

        nelbo = -1.0 * (tf.reduce_sum(input_tensor=expected_log_terms) *
                        (self.total_training_data / tf.cast(n_train, settings.tf_float_type)) - tf.reduce_sum(input_tensor=KL_sums))

        return nelbo

    @tf.function
    def call(self, inputs):
        """
        @return: Predictive mean and variance of the GP model
        """

        if not self.initialized:
            self._init_parameters()

        matrix_params = self.update_params(inputs)
        
        mean_pred, var_pred = self._build_predictive_gaussian_mean_vars(inputs, matrix_params)

        # y_train was standardize so we need to back transform the predictions
        mean_pred = mean_pred * self.y_std + self.y_mean
        var_pred *= self.y_std**2

        return mean_pred, var_pred


    @tf.function
    def _build_KL_objective(self, chol_Kmm, m, L):

        alpha = tf.linalg.triangular_solve(chol_Kmm, m)
        # Mahalanobis term: \mu_q^{t} \sigma_p^{-1} \mu_q
        mahalanobis = tf.reduce_sum(input_tensor=tf.square(alpha), axis=[-1, -2])

        # diagonal \sigma_p and \sigma_q
        Lq_diag = tf.linalg.diag_part(L)
        Lp_diag = tf.linalg.diag_part(chol_Kmm)

        # Constant term: - N
        constant = - tf.constant(self.num_inducing, settings.tf_float_type)

        # Log-determinant of the covariance of q(x):
        logdet_qcov = 2.0 * tf.reduce_sum(input_tensor=tf.math.log( Lq_diag ), axis=[-1])
        # Log-determinant of the covariance of p(x):
        logdet_pcov = 2.0 * tf.reduce_sum(input_tensor=tf.math.log( Lp_diag ), axis=[-1])

        # Trace term: tr(\sigma_p^{-1} \sigma_q)
        trace = tf.reduce_sum(input_tensor=tf.square(tf.linalg.triangular_solve(chol_Kmm, L)) , axis=[-1, -2])

        KL_sum = 0.5*(constant  + trace + mahalanobis + (logdet_pcov - logdet_qcov))
            
        return tf.reduce_sum(input_tensor=KL_sum)
        
    @tf.function
    def _build_mean_vars(self, inputs, matrix_params):
        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # We get the marginals of the values of the process
        
        means = tf.squeeze(tf.matmul(Knm, KmmInvm))
         
        marginal_variances = self.kernel.get_var_points(inputs)
        
        variances = - tf.reduce_sum(input_tensor=KmmInvKmn * tf.transpose(a=Knm), axis=0) + \
                tf.reduce_sum(input_tensor=LtKmmInvKmn * LtKmmInvKmn, axis=0) + marginal_variances

        return means, variances

    @tf.function
    def _build_expected_log_term(self, inputs, outputs, matrix_params):
        '''
        Computes E_q log p(y|f) 
        @param data_inputs: input data
        @param data_targets: output data
        @return:
        '''

        means, variances = self._build_mean_vars(inputs, matrix_params)

        return self.likelihood.variational_expectations(means, variances, outputs)

    @tf.function
    def _build_predictive_gaussian_mean_vars(self, inputs, matrix_params):
        '''
        Computes the mean and variance of E_q p(y|f) 
        '''

        means, variances = self._build_mean_vars(inputs, matrix_params)
        
        return self.likelihood(means, variances)

    
    def get_joint_prediction(self, inputs):
        
        matrix_params = self.update_params(inputs)

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # means = tf.squeeze(tf.matmul(Knm, KmmInvm))
        means = tf.reshape(tf.matmul(Knm, KmmInvm), (-1, 1))

        Kstar = self.kernel(inputs)
        KnmKmmInvKmn = tf.matmul(Knm, KmmInvKmn)
        KnmKmmInvSKmmInvKmn = tf.squeeze(tf.matmul(LtKmmInvKmn, LtKmmInvKmn, transpose_a=True))

        covariances = - KnmKmmInvKmn + KnmKmmInvSKmmInvKmn + Kstar

        return means, covariances


class SVGP_Hensman(SVGP_Titsias):
    '''
    Variational Inference regression with sparse gaussian processes
    '''
    
    def __init__(self, kernel, likelihood, inducing_points, total_training_data, input_dim, y_mean, y_std, path_results='', seed=0):

        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood used for the model
        @param inducing_points: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        '''
        super(SVGP_Hensman, self).__init__(kernel, likelihood, inducing_points, total_training_data, input_dim, None, None, y_mean, y_std, path_results, seed)


    def _init_parameters(self):
        """
        Initializes the trainable parameters of the model
        """

        # update Kernels and matrix computation
        self.inducing_points = tf.Variable(self.inducing_points, dtype = settings.tf_float_type, name="inducing_points")
        self.m = tf.Variable(tf.zeros([ self.num_inducing, 1 ], dtype = settings.tf_float_type), name="m")
        Kmm = self.kernel(self.inducing_points)  # K_{mm} in the paper
        chol_Kmm_scaled = tf.linalg.cholesky(Kmm) * 1e-5 # K_{mm} = LL^{t}
        self.Lraw = tf.Variable(chol_Kmm_scaled - tf.linalg.band_part(chol_Kmm_scaled, 0, 0) + \
                        tf.linalg.diag(tf.math.log(tf.linalg.diag_part(chol_Kmm_scaled))), dtype = settings.tf_float_type, name="Lraw")

        self.initialized = True
        
    @tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """
        Knm = self.kernel(inputs, self.inducing_points)  # K_{nm}
        Kmm = self.kernel(self.inducing_points)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}
        L = tf.linalg.band_part(self.Lraw, -1, 0) - tf.linalg.band_part(self.Lraw, 0, 0) + \
               tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Lraw)))

        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, self.m)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(a=Knm))
        LtKmmInvKmn = tf.matmul(tf.transpose(a=L), KmmInvKmn)

        return (Knm, chol_Kmm, self.m, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)



class SVGP_NN(SVGP_Hensman):
    '''
    Variational Inference regression with sparse gaussian processes generating the inducing points and q from NNs
    '''
    
    def __init__(self, kernel, likelihood, num_inducing_points, total_training_data, input_dim, y_mean, y_std, n_hidden1 = 15, n_layers1 = 2, path_results='', dropout_rate = 0.5, seed=0):

        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood for the model
        @param num_inducing_points: Number of inducing points
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        @param n_hidden1: Number of hidden units of the neural network that generates Z
        @param n_layers1: Number of layers of the neural network that generates Z
        @param n_hidden2: Number of hidden units of the neural network that generates q
        @param n_layers2: Number of layers of the neural network that generates q
        '''
        inducing_points = np.zeros([num_inducing_points])
        
        super(SVGP_NN, self).__init__(kernel, likelihood, inducing_points, total_training_data, input_dim, y_mean, y_std, path_results, seed)

        self._create_nns(input_dim, n_layers1, n_hidden1, dropout_rate = dropout_rate, batch_normalization = True, soft_ini = False)

    def _create_nns(self, input_dim, n_layers1, n_hidden1, dropout_rate, batch_normalization = False, soft_ini = False):
        """
        Create two NN: One for the inducing points and other for q
        """

        activation = "sigmoid"

        if soft_ini:
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3, seed = self.seed)
            initializer_last_bias = tf.keras.initializers.RandomNormal(mean=-10, stddev=1e-3, seed = self.seed)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed = self.seed)
            initializer_last_bias = "zeros"

        dim_out = int(self.num_inducing*(self.num_inducing - 1)/2 + 2*self.num_inducing)+ self.num_inducing*input_dim
        # Define the two NN
        ## NN that generates the Inducing points
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        self.net = tf.keras.layers.Dense(n_hidden1, kernel_initializer=initializer, dtype = self.dtype)(input_layer)

        if n_layers1 > 1 and batch_normalization:
            self.net = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, dtype = self.dtype)(self.net)
        self.net = tf.keras.layers.Activation(activation, dtype=self.dtype)(self.net)
        for _ in range(n_layers1 - 1):
            self.net = tf.keras.layers.Dense(n_hidden1, kernel_initializer=initializer, dtype = self.dtype)(self.net)
            if batch_normalization:
                self.net = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, dtype = self.dtype)(self.net)
            self.net = tf.keras.layers.Activation(activation, dtype=self.dtype)(self.net)
        self.net = tf.keras.layers.Dense(dim_out, kernel_initializer=initializer, bias_initializer = initializer_last_bias,  dtype = self.dtype)(self.net)
        self.net = tf.keras.layers.Dropout(rate=dropout_rate, dtype=self.dtype)(self.net)
        if soft_ini:
            self.net = tf.keras.layers.Add(dtype = self.dtype)([tf.tile(input_layer, [1, self.num_inducing]), self.net])

        self.net = tf.keras.Model(input_layer, self.net, name="Net1")

    def _init_parameters(self):
        """
        Initializes the trainable parameters of the model
        """

        self.initialized = True

    def _get_output_nn(self, inputs):
        """
        Returns the output of the NN
        """
        return self.net(inputs)
        
    @tf.function
    def update_params(self, inputs, full_cov=False):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """
        # update Kernels and matrix computation
        output_nets = self._get_output_nn(inputs)
        output_net1 = output_nets[:,:self.num_inducing * self.input_dim]
        inducing_points = tf.reshape(output_net1, [tf.shape(input=output_net1)[0], self.num_inducing, self.input_dim])

        m = output_nets[:, self.num_inducing * self.input_dim:self.num_inducing * self.input_dim+self.num_inducing]

        sub_out = output_nets[:, self.num_inducing * self.input_dim+self.num_inducing:]
        Lraw = tfp.math.fill_triangular(sub_out)
        m_expanded = tf.expand_dims(m, -1)

        if full_cov:
            input_tiled = tf.tile(inputs[None,:,:], [inducing_points.shape[0], 1, 1])
        else:
            input_tiled = inputs

        Knm = self.kernel(input_tiled, inducing_points)

        Kmm = self.kernel(inducing_points)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}
        L = tf.linalg.band_part(Lraw, -1, 0) - tf.linalg.band_part(Lraw, 0, 0) + \
               tf.linalg.diag(tf.exp(tf.linalg.diag_part(Lraw)))

        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, m_expanded)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(a=Knm, perm=[0, 2, 1]))
        LtKmmInvKmn = tf.matmul(L, KmmInvKmn, transpose_a = True)

        return (Knm, chol_Kmm, m_expanded, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)

    @tf.function
    def _build_mean_vars(self, inputs, matrix_params):

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # We get the marginals of the values of the process

        means = tf.squeeze(tf.matmul(Knm, KmmInvm))

        marginal_variances = self.kernel.get_var_points(inputs)
        marginal_variances = tf.expand_dims(marginal_variances, -1)

        variances = - tf.reduce_sum(input_tensor=KmmInvKmn * tf.transpose(a=Knm, perm=[0, 2, 1]), axis=[1]) + \
                tf.reduce_sum(input_tensor=LtKmmInvKmn * LtKmmInvKmn, axis=[1]) + marginal_variances

        return means, tf.squeeze(variances)
        
    @tf.function
    def _build_KL_objective(self, chol_Kmm, m, L):

        KL_sum = super(SVGP_NN, self)._build_KL_objective(chol_Kmm, m, L)
            
        return KL_sum / tf.cast(tf.shape(m)[0], self.dtype)


    def get_joint_prediction(self, inputs):

        matrix_params = self.update_params(inputs, full_cov=False)

        means, vars = self._build_mean_vars(inputs, matrix_params)
        means = tf.reshape(means, (-1, 1))

        matrix_params = self.update_params(inputs, full_cov=True)

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # means = tf.linalg.diag_part(tf.matmul(Knm, KmmInvm)) 
        # Mean of means
        # means = tf.reduce_mean(tf.matmul(Knm, KmmInvm), axis = 0) 

        Kstar = self.kernel(inputs)
        KnmKmmInvKmn = tf.matmul(Knm, KmmInvKmn)
        KnmKmmInvSKmmInvKmn = tf.squeeze(tf.matmul(LtKmmInvKmn, LtKmmInvKmn, transpose_a=True))

        covariance = - KnmKmmInvKmn + KnmKmmInvSKmmInvKmn + Kstar
    
        # covariances = tf.reduce_mean(covariance, axis=0)
        # covariances = covariances - tf.linalg.band_part(covariances, 0, 0) + tf.linalg.diag(vars)
        # Mean of covariances
        covariances = tf.reduce_mean(covariance, axis=0)

        return means, covariances
        

class SVGP_NNJ(SVGP_NN):
    '''
    Variational Inference regression with sparse gaussian processes generating the inducing points and q from NNs
    Here the NN outputs a set of inducing points per minibatch
    '''
    
    def __init__(self, kernel, likelihood, num_inducing_points, total_training_data, batch_size, input_dim, y_mean, y_std, n_hidden1 = 15, n_layers1 = 2, path_results='', dropout_rate = 0.5, seed=0):

        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood for the model
        @param num_inducing_points: Number of inducing points
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        @param n_hidden1: Number of hidden units of the neural network that generates Z
        @param n_layers1: Number of layers of the neural network that generates Z
        @param n_hidden2: Number of hidden units of the neural network that generates q
        @param n_layers2: Number of layers of the neural network that generates q
        '''

        self.batch_size = batch_size

        super(SVGP_NNJ, self).__init__(kernel, likelihood, num_inducing_points, total_training_data, input_dim, y_mean, y_std, n_hidden1, n_layers1, path_results, dropout_rate, seed)
                                    

    def _create_nns(self, input_dim, n_layers1, n_hidden1, dropout_rate, batch_normalization = False, soft_ini = False):
        """
        Create two NN: One for the inducing points and other for q
        """

        activation = "sigmoid"

        if soft_ini:
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3, seed = self.seed)
            initializer_last_bias = tf.keras.initializers.RandomNormal(mean=-10, stddev=1e-3, seed = self.seed)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed = self.seed)
            initializer_last_bias = "zeros"

        dim_out = int(self.num_inducing*(self.num_inducing - 1)/2 + 2*self.num_inducing)+ self.num_inducing*input_dim
        # Define the two NN
        ## NN that generates the Inducing points
        input_layer = tf.keras.layers.Input(shape=(input_dim * self.batch_size,))
        self.net = tf.keras.layers.Dense(n_hidden1, kernel_initializer=initializer, dtype = self.dtype)(input_layer)

        if n_layers1 > 1 and batch_normalization:
            self.net = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, dtype = self.dtype)(self.net)
        self.net = tf.keras.layers.Activation(activation, dtype=self.dtype)(self.net)
        for _ in range(n_layers1 - 1):
            self.net = tf.keras.layers.Dense(n_hidden1, kernel_initializer=initializer, dtype = self.dtype)(self.net)
            if batch_normalization:
                self.net = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, dtype = self.dtype)(self.net)
            self.net = tf.keras.layers.Activation(activation, dtype=self.dtype)(self.net)
        self.net = tf.keras.layers.Dense(dim_out, kernel_initializer=initializer, bias_initializer = initializer_last_bias,  dtype = self.dtype)(self.net)
        self.net = tf.keras.layers.Dropout(rate=dropout_rate, dtype=self.dtype)(self.net)
        if soft_ini:
            self.net = tf.keras.layers.Add(dtype = self.dtype)([tf.tile(input_layer, [1, self.num_inducing]), self.net])

        self.net = tf.keras.Model(input_layer, self.net, name="Net1")

 
    def _get_output_nn(self, inputs):
        """
        Returns the output of the NN
        """
        return self.net(tf.reshape(inputs, (1,-1)))
        

    @tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """
        # update Kernels and matrix computation
        output_nets = self._get_output_nn(inputs)
        output_net1 = output_nets[:,:self.num_inducing * self.input_dim]
        inducing_points = tf.reshape(output_net1, [self.num_inducing, self.input_dim])

        m = output_nets[:, self.num_inducing * self.input_dim:self.num_inducing * self.input_dim+self.num_inducing]

        sub_out = output_nets[:, self.num_inducing * self.input_dim+self.num_inducing:]
        Lraw = tfp.math.fill_triangular(sub_out)
        m_expanded = tf.expand_dims(m, -1)

        Knm = self.kernel(inputs, inducing_points)  # K_{nm}
        Kmm = self.kernel(inducing_points)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}
        L = tf.linalg.band_part(Lraw, -1, 0) - tf.linalg.band_part(Lraw, 0, 0) + \
               tf.linalg.diag(tf.exp(tf.linalg.diag_part(Lraw)))

        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, m_expanded)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(Knm))
        LtKmmInvKmn = tf.matmul(L, KmmInvKmn, transpose_a = True)

        return (Knm, chol_Kmm, m_expanded, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)


    @tf.function
    def _build_mean_vars(self, inputs, matrix_params):

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # We get the marginals of the values of the process

        means = tf.squeeze(tf.matmul(Knm, KmmInvm))

        marginal_variances = self.kernel.get_var_points(inputs)

        variances = - tf.reduce_sum(input_tensor=KmmInvKmn * tf.transpose(Knm), axis=[0]) + \
                tf.reduce_sum(input_tensor=LtKmmInvKmn * LtKmmInvKmn, axis=[1]) + marginal_variances

        return means, tf.squeeze(variances)

    def get_joint_prediction(self, inputs):
        
        matrix_params = self.update_params(inputs)

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        means = tf.reshape(tf.matmul(Knm, KmmInvm), (-1, 1))
        

        Kstar = self.kernel(inputs)
        KnmKmmInvKmn = tf.matmul(Knm, KmmInvKmn)
        KnmKmmInvSKmmInvKmn = tf.squeeze(tf.matmul(LtKmmInvKmn, LtKmmInvKmn, transpose_a=True))

        covariances = - KnmKmmInvKmn + KnmKmmInvSKmmInvKmn + Kstar

        return means, covariances
        
class SVGP_NNU(SVGP_NNJ):
    '''
    Variational Inference regression with sparse gaussian processes generating the inducing points and q from NNs
    Here we consider the union of the inducing points that the NN outputs
    '''
    
    def __init__(self, kernel, likelihood, num_inducing_points, total_training_data, batch_size, input_dim, y_mean, y_std, n_hidden1 = 15, n_layers1 = 2, path_results='', dropout_rate = 0.5, seed=0):

        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood for the model
        @param num_inducing_points: Number of inducing points
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        @param n_hidden1: Number of hidden units of the neural network that generates Z
        @param n_layers1: Number of layers of the neural network that generates Z
        @param n_hidden2: Number of hidden units of the neural network that generates q
        @param n_layers2: Number of layers of the neural network that generates q
        '''

        super(SVGP_NNU, self).__init__(kernel, likelihood, num_inducing_points, total_training_data, batch_size, input_dim, y_mean, y_std, n_hidden1, n_layers1, path_results, dropout_rate, seed)


    def _create_nns(self, input_dim, n_layers1, n_hidden1, dropout_rate, batch_normalization = False, soft_ini = False):
        """
        Create two NN: One for the inducing points and other for q
        """

        activation = "sigmoid"

        if soft_ini:
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3, seed = self.seed)
            initializer_last_bias = tf.keras.initializers.RandomNormal(mean=-10, stddev=1e-3, seed = self.seed)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed = self.seed)
            initializer_last_bias = "zeros"

        effective_inducing = self.num_inducing * self.batch_size
        dim_out = (int((effective_inducing*(effective_inducing - 1)/2) + effective_inducing) / self.batch_size + self.num_inducing)+ self.num_inducing*input_dim
        # Define the two NN
        ## NN that generates the Inducing points
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        self.net = tf.keras.layers.Dense(n_hidden1, kernel_initializer=initializer, dtype = self.dtype)(input_layer)

        if n_layers1 > 1 and batch_normalization:
            self.net = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, dtype = self.dtype)(self.net)
        self.net = tf.keras.layers.Activation(activation, dtype=self.dtype)(self.net)
        for _ in range(n_layers1 - 1):
            self.net = tf.keras.layers.Dense(n_hidden1, kernel_initializer=initializer, dtype = self.dtype)(self.net)
            if batch_normalization:
                self.net = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, dtype = self.dtype)(self.net)
            self.net = tf.keras.layers.Activation(activation, dtype=self.dtype)(self.net)
        self.net = tf.keras.layers.Dense(dim_out, kernel_initializer=initializer, bias_initializer = initializer_last_bias,  dtype = self.dtype)(self.net)
        self.net = tf.keras.layers.Dropout(rate=dropout_rate, dtype=self.dtype)(self.net)
        if soft_ini:
            self.net = tf.keras.layers.Add(dtype = self.dtype)([tf.tile(input_layer, [1, self.num_inducing]), self.net])

        self.net = tf.keras.Model(input_layer, self.net, name="Net1")


    def _get_output_nn(self, inputs):
        """
        Returns the output of the NN
        """
        return self.net(inputs)


    @tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """
        # update Kernels and matrix computation
        output_nets = self._get_output_nn(inputs)
        output_net1 = output_nets[:,:self.num_inducing * self.input_dim]
        inducing_points = tf.reshape(output_net1, [tf.shape(input=output_net1)[0] * self.num_inducing, self.input_dim])

        output_net2 = output_nets[:, self.num_inducing * self.input_dim:self.num_inducing * self.input_dim+self.num_inducing]
        m = tf.reshape(output_net2,  (-1, 1))

        sub_out = output_nets[:, self.num_inducing * self.input_dim+self.num_inducing:]

        Lraw = tfp.math.fill_triangular(tf.reshape(sub_out, (1,-1))[0])

        Knm = self.kernel(inputs, inducing_points)  # K_{nm}
        Kmm = self.kernel(inducing_points)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}
        L = tf.linalg.band_part(Lraw, -1, 0) - tf.linalg.band_part(Lraw, 0, 0) + \
               tf.linalg.diag(tf.exp(tf.linalg.diag_part(Lraw)))

        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, m)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(Knm))
        LtKmmInvKmn = tf.matmul(L, KmmInvKmn, transpose_a = True)

        return (Knm, chol_Kmm, m, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)


    @tf.function
    def _build_mean_vars(self, inputs, matrix_params):

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # We get the marginals of the values of the process

        means = tf.squeeze(tf.matmul(Knm, KmmInvm))

        marginal_variances = self.kernel.get_var_points(inputs)

        variances = - tf.reduce_sum(input_tensor=KmmInvKmn * tf.transpose(Knm), axis=[0]) + \
                tf.reduce_sum(input_tensor=LtKmmInvKmn * LtKmmInvKmn, axis=[0]) + marginal_variances

        return means, tf.squeeze(variances)
        


class SWSGP(SVGP_Hensman):
    '''
    Sparse within Sparse Gaussian Processes using Neighbor Information
    https://arxiv.org/pdf/2011.05041.pdf
    '''

    def __init__(self, kernel, likelihood, inducing_points, num_inducing_closest, total_training_data, input_dim,
                 y_mean, y_std, n_hidden1=15, n_layers1=2, n_hidden2=15, n_layers2=2, path_results='', seed=0):
        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood for the model
        @param inducing_points: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param num_inducing_points: Number of inducing points to select for each minibatch
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        @param n_hidden1: Number of hidden units of the neural network that generates Z
        @param n_layers1: Number of layers of the neural network that generates Z
        @param n_hidden2: Number of hidden units of the neural network that generates q
        @param n_layers2: Number of layers of the neural network that generates q
        '''

        super(SWSGP, self).__init__(kernel, likelihood, inducing_points, total_training_data, input_dim, y_mean, y_std,
                                    path_results, seed)

        self.num_inducing_closest = num_inducing_closest
        


    #@tf.function
    def _compute_distance(self, x1, x2):
        """
        Computes Euclidean distances between x1 and x2
        Args:
          x1,    [m,d] matrix
          x2,    [n,d] matrix
        Returns:
          covar,    [m] Euclidean distances
        """

        value = tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(x1), axis=-1), -1)
        value2 = tf.transpose(a=tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(x2), axis=-1), -1))

        covar = tf.maximum(value - 2 * tf.matmul(x1, x2, False, True) + value2, 0.0)
        covar = tf.sqrt(tf.maximum(covar, 1e-40))

        return covar


    #@tf.function
    def select_inducing_points(self, inputs):
        """
        Select the closest inducing points to the inputs
        @param inputs: Minibatch of points
        """
        batch_size = tf.shape(inputs)[0]

        ############### Compute Indexes ###################
        distances = self._compute_distance(inputs, self.inducing_points)

        _, closest_indices = tf.nn.top_k(-distances, self.num_inducing_closest, sorted = False)
        closest_indices = tf.sort(closest_indices, axis=1)
        row_indices = tf.repeat(tf.reshape(tf.range(0, batch_size), [-1, 1]), [self.num_inducing_closest], axis = 1)
        selected_indices = tf.stack([row_indices, closest_indices], axis=2)

        ############### M ###################
        m_rep = tf.repeat(tf.expand_dims(self.m, 0), batch_size, axis = 0)
        selected_m = tf.gather_nd(m_rep, selected_indices)

        ############### IP ###################
        ip_rep = tf.repeat(tf.expand_dims(self.inducing_points, 0), batch_size, axis=0)
        selected_ip = tf.gather_nd(ip_rep, selected_indices)

        ############### L ###################
        #L = tf.linalg.band_part(self.Lraw, -1, 0) - tf.linalg.band_part(self.Lraw, 0, 0) + \
                   #  tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Lraw)))
        L_rep = tf.repeat(tf.expand_dims(self.Lraw, 0), batch_size, axis=0)
        L_sl = tf.gather_nd(L_rep, selected_indices)
        selected_L = tf.transpose(tf.gather_nd(tf.transpose(L_sl, [0, 2, 1]), selected_indices), [0, 2, 1])
        selected_L = tf.linalg.band_part(selected_L, -1, 0) - tf.linalg.band_part(selected_L, 0, 0) + \
                     tf.linalg.diag(tf.exp(tf.linalg.diag_part(selected_L)))

        return selected_ip, selected_m, selected_L

    #@tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """

        inducing_points, m_expanded, L = self.select_inducing_points(inputs)

        #m_expanded = tf.expand_dims(m, -1)

        Knm = self.kernel(inputs, inducing_points)  # K_{nm}
        Kmm = self.kernel(inducing_points)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}

        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, m_expanded)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(a=Knm, perm=[0, 2, 1]))
        LtKmmInvKmn = tf.matmul(L, KmmInvKmn, transpose_a = True)

        return (Knm, chol_Kmm, m_expanded, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)


    #@tf.function
    def _build_mean_vars(self, inputs, matrix_params):

        Knm, _, _, _, KmmInvm, KmmInvKmn, LtKmmInvKmn = matrix_params

        # We get the marginals of the values of the process

        means = tf.squeeze(tf.matmul(Knm, KmmInvm))

        marginal_variances = self.kernel.get_var_points(inputs)
        marginal_variances = tf.expand_dims(marginal_variances, -1)

        variances = - tf.reduce_sum(input_tensor=KmmInvKmn * tf.transpose(a=Knm, perm=[0, 2, 1]), axis=[1]) + \
                tf.reduce_sum(input_tensor=LtKmmInvKmn * LtKmmInvKmn, axis=[1]) + marginal_variances

        return means, tf.squeeze(variances)

    @tf.function
    def _build_KL_objective(self, chol_Kmm, m, L):

        KL_sum = super(SWSGP, self)._build_KL_objective(chol_Kmm, m, L)
            
        return KL_sum / tf.cast(tf.shape(m)[0], self.dtype)


    def get_joint_prediction(self, inputs):
        raise NotImplementedError("This method cannot compute covariates.")


class SWSGPU(SVGP_Hensman):

    def __init__(self, kernel, likelihood, inducing_points, num_inducing_closest, total_training_data, input_dim,
                 y_mean, y_std, n_hidden1=15, n_layers1=2, n_hidden2=15, n_layers2=2, path_results='', seed=0):
        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood for the model
        @param inducing_points: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param num_inducing_points: Number of inducing points to select for each minibatch
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        @param n_hidden1: Number of hidden units of the neural network that generates Z
        @param n_layers1: Number of layers of the neural network that generates Z
        @param n_hidden2: Number of hidden units of the neural network that generates q
        @param n_layers2: Number of layers of the neural network that generates q
        '''
        super(SWSGPU, self).__init__(kernel, likelihood, inducing_points, total_training_data, input_dim, y_mean, y_std,
                                    path_results, seed)

        self.num_inducing_closest = num_inducing_closest


    def _compute_distance(self, x1, x2):
        """
        Computes Euclidean distances between x1 and x2
        Args:
          x1,    [m,d] matrix
          x2,    [n,d] matrix
        Returns:
          covar,    [m] Euclidean distances
        """

        value = tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(x1), axis=-1), -1)
        value2 = tf.transpose(a=tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(x2), axis=-1), -1))

        covar = tf.maximum(value - 2 * tf.matmul(x1, x2, False, True) + value2, 0.0)
        covar = tf.sqrt(tf.maximum(covar, 1e-40))

        return covar


    def select_inducing_points(self, inputs):
        """
        Select the closest inducing points to the inputs
        @param inputs: Minibatch of points
        """

        ############### Compute Indexes ###################
        distances = self._compute_distance(inputs, self.inducing_points)
        _, closest_indices = tf.nn.top_k(-distances, self.num_inducing_closest, sorted = False)

        # Select the union (remove duplicates)
        selected_indices = tf.unique(tf.reshape(closest_indices, (-1, 1))[:,0]).y
        
        ############### M ###################
        selected_m = tf.gather(self.m, selected_indices)

        ############### IP ###################
        selected_ip = tf.gather(self.inducing_points, selected_indices)

        ############### L ###################
        selected_L = tf.gather(tf.gather(self.Lraw, selected_indices), selected_indices, axis=1)
        selected_L = tf.linalg.band_part(selected_L, -1, 0) - tf.linalg.band_part(selected_L, 0, 0) + \
                     tf.linalg.diag(tf.exp(tf.linalg.diag_part(selected_L)))

        return selected_ip, selected_m, selected_L


    @tf.function
    def update_params(self, inputs):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """

        selected_ip, m, L = self.select_inducing_points(inputs)

        Knm = self.kernel(inputs, selected_ip)  # K_{nm}
        Kmm = self.kernel(selected_ip)  # K_{mm} in the paper
        chol_Kmm = tf.linalg.cholesky(Kmm) # K_{mm} = LL^{t}

        KmmInvm = tf.linalg.cholesky_solve(chol_Kmm, m)

        KmmInvKmn = tf.linalg.cholesky_solve(chol_Kmm, tf.transpose(a=Knm))
        LtKmmInvKmn = tf.matmul(tf.transpose(a=L), KmmInvKmn)

        return (Knm, chol_Kmm, m, L, KmmInvm, KmmInvKmn, LtKmmInvKmn)



class SVGP_SOLVE(SVGP):
    """
    SOLVE-GP: Sparse Orthogonal Variational Inference for Gaussian Processes
    """

    def __init__(self, kernel, likelihood, inducing_points_u, inducing_points_v, total_training_data, input_dim, y_mean, y_std,
                 path_results='', seed = 0):

        '''
        Constructor
        @param kernel: Kernel (covariance) functions
        @param likelihood: Likelihood used for the model
        @param inducing_points_u: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param inducing_points_v: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param total_training_data : Use to scale the objective
        @param y_mean: mean of the targets (we assume standardized targets)
        @param y_std: std of the targets (we assume standardized targets)
        '''
        super(SVGP_SOLVE, self).__init__('SVGP_SOLVE', likelihood, y_mean, y_std, seed, dtype=settings.tf_float_type)

        self.path_results = path_results
        self.kernel = kernel

        # number of training points
        self.total_training_data = tf.constant([1.0 * total_training_data], dtype=settings.tf_float_type)

        # Repeat the inducing inputs for all latent processes if we haven't been given individually
        # specified inputs per process.

        self.input_dim = input_dim  # input dimension
        self.inducing_points_u = inducing_points_u
        self.num_ip_u = inducing_points_u.shape[0]
        self.inducing_points_v = inducing_points_v
        self.num_ip_v = inducing_points_v.shape[0]

        self.initialized = False

    def _init_parameters(self):
        """
        Initializes the trainable parameters of the model
        """

        # update Kernels and matrix computation
        self.a = tf.Variable(self.inducing_points_u, dtype=settings.tf_float_type, name="inducing_points_a")
        self.z = tf.Variable(self.inducing_points_v, dtype=settings.tf_float_type, name="inducing_points_z")

        self.qu_m = tf.Variable(tf.zeros([self.num_ip_u, 1], dtype=settings.tf_float_type), name="qu_m")

        Kuu = self.kernel(self.z)  # K_{uu}
        chol_Kuu = tf.linalg.cholesky(Kuu)
        chol_Kuu_scaled = chol_Kuu * 1e-5  # K_{mm} = LL^{t}
        self.Lraw_uu = tf.Variable(chol_Kuu_scaled - tf.linalg.band_part(chol_Kuu_scaled, 0, 0) + \
                                tf.linalg.diag(tf.math.log(tf.linalg.diag_part(chol_Kuu_scaled))),
                                dtype=settings.tf_float_type, name="Lraw_uu")



        # self.qv_m = tf.zeros([self.num_ip_v, 1], dtype=settings.tf_float_type)
        self.qv_m = tf.Variable(tf.zeros([self.num_ip_v, 1], dtype=settings.tf_float_type), name="qv_m")

        Kvv = self.kernel(self.a)  # K_{uu}
        Kuv = self.kernel(self.z, self.a)

        Lu_inv_Kuv = tf.linalg.triangular_solve(chol_Kuu, Kuv)

        Cvv = Kvv - tf.matmul(
            Lu_inv_Kuv, Lu_inv_Kuv, transpose_a=True)

        chol_Cvv_scaled = tf.linalg.cholesky(Cvv) * 1e-5  # Cvv = LL^{t}
        self.Lraw_vv = tf.Variable(chol_Cvv_scaled - tf.linalg.band_part(chol_Cvv_scaled, 0, 0) + \
                                tf.linalg.diag(tf.math.log(tf.linalg.diag_part(chol_Cvv_scaled))),
                                dtype=settings.tf_float_type, name="Lraw_vv")

        self.initialized = True

    # @tf.function
    def update_params(self, inputs, full_cov=False):
        """
        @param x: input
        @return: Updates all the matrices that will be needed to obtain the variationnal
            lower bound.
        """


        Kuu = self.kernel(self.z)

        Kvv = self.kernel(self.a)

        chol_Kuu = tf.linalg.cholesky(Kuu)

        Kuv = self.kernel(self.z, self.a)

        CholKuu_inv_Kuv = tf.linalg.triangular_solve(chol_Kuu, Kuv)

        Cvv = Kvv -tf.matmul( CholKuu_inv_Kuv, CholKuu_inv_Kuv, transpose_a=True)

        chol_Kvv = tf.linalg.cholesky(Cvv)


        L_u = tf.linalg.band_part(self.Lraw_uu, -1, 0) - tf.linalg.band_part(self.Lraw_uu, 0, 0) + \
            tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Lraw_uu)))

        # L_v = chol_Kvv
        L_v = tf.linalg.band_part(self.Lraw_vv, -1, 0) - tf.linalg.band_part(self.Lraw_vv, 0, 0) + \
            tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Lraw_vv)))

        Kun = self.kernel(self.z, inputs)  # K_{nm}
        Kvn = self.kernel(self.a, inputs)
        chol_KuuInvKun = tf.linalg.triangular_solve(chol_Kuu, Kun)
        KuuInvKun = tf.linalg.triangular_solve(tf.transpose(chol_Kuu), chol_KuuInvKun, lower=False)

        Cvn = Kvn - tf.matmul(CholKuu_inv_Kuv, chol_KuuInvKun, transpose_a=True)

        if full_cov:
            Kff = self.kernel(inputs)
            Cff = Kff - tf.matmul(chol_KuuInvKun, chol_KuuInvKun, transpose_a=True)
        else:
            Kff = self.kernel.get_var_points(inputs)
            Cff = Kff - tf.reduce_sum(tf.square(chol_KuuInvKun), axis=0)

        chol_KvvInv_Cvn = tf.linalg.triangular_solve(chol_Kvv, Cvn)

        KvvInvCun = tf.linalg.triangular_solve(tf.transpose(chol_Kvv), chol_KvvInv_Cvn, lower=False)


        # [K, M, M]
        Su = tf.linalg.band_part(L_u, -1, 0)
        SuTA = tf.matmul(Su, KuuInvKun, transpose_a=True)


        Sv = tf.linalg.band_part(L_v, -1, 0)
        SvTA = tf.matmul(Sv, KvvInvCun, transpose_a=True)

        return [KuuInvKun, KvvInvCun, SuTA, SvTA, chol_KvvInv_Cvn, Cff, L_u, L_v, chol_Kuu, chol_Kvv]



    # @tf.function
    def nelbo(self, inputs, outputs):
        """
        @return: NELBO for the current model
        """

        if not self.initialized:
            self._init_parameters()

        matrix_params = self.update_params(inputs)

        [_, _, _, _, _, _, L_u, L_v, Kuu, Cvv] = matrix_params

        KL_u_sum = self._build_KL_objective(Kuu, self.qu_m, L_u, self.num_ip_u)
        KL_v_sum = self._build_KL_objective(Cvv, self.qv_m, L_v, self.num_ip_v)
        KL_sums = KL_u_sum + KL_v_sum

        expected_log_terms = self._build_expected_log_term(outputs, matrix_params)

        n_train = tf.shape(input=inputs)[0]

        nelbo = -1.0 * (tf.reduce_sum(input_tensor=expected_log_terms) *
                        (self.total_training_data / tf.cast(n_train, settings.tf_float_type)) - tf.reduce_sum(
                    input_tensor=KL_sums))

        return nelbo

    # @tf.function
    def call(self, inputs):
        """
        @return: Predictive mean and variance of the GP model
        """

        if not self.initialized:
            self._init_parameters()

        matrix_params = self.update_params(inputs)

        mean_pred, var_pred = self._build_predictive_gaussian_mean_vars(matrix_params)

        # y_train was standardize so we need to back transform the predictions
        mean_pred = mean_pred * self.y_std + self.y_mean
        var_pred *= self.y_std ** 2

        return mean_pred, var_pred

    # @tf.function
    def _build_KL_objective(self, chol_Kmm, m, L, num_inducing_points):

        alpha = tf.linalg.triangular_solve(chol_Kmm, m)
        # Mahalanobis term: \mu_q^{t} \sigma_p^{-1} \mu_q
        mahalanobis = tf.reduce_sum(input_tensor=tf.square(alpha), axis=[-1, -2])

        # diagonal \sigma_p and \sigma_q
        Lq_diag = tf.linalg.diag_part(L)
        Lp_diag = tf.linalg.diag_part(chol_Kmm)

        # Constant term: - N
        constant = - tf.constant(num_inducing_points, settings.tf_float_type)

        # Log-determinant of the covariance of q(x):
        logdet_qcov = 2.0 * tf.reduce_sum(input_tensor=tf.math.log(Lq_diag), axis=[-1])
        # Log-determinant of the covariance of p(x):
        logdet_pcov = 2.0 * tf.reduce_sum(input_tensor=tf.math.log(Lp_diag), axis=[-1])

        # Trace term: tr(\sigma_p^{-1} \sigma_q)
        trace = tf.reduce_sum(input_tensor=tf.square(tf.linalg.triangular_solve(chol_Kmm, L)), axis=[-1, -2])

        KL_sum = 0.5 * (constant + trace + mahalanobis + (logdet_pcov - logdet_qcov))

        return tf.reduce_sum(input_tensor=KL_sum)

    # @tf.function
    def _build_mean_vars(self, matrix_params, full_cov=False):

        KuuInvKun, KvvInvCun, SuTA, SvTA, chol_KvvInv_Cvn, Cff, _, _, _, _ = matrix_params
        fu = tf.matmul(KuuInvKun, self.qu_m, transpose_a=True)
        fv = tf.matmul(KvvInvCun, self.qv_m, transpose_a=True)

        if full_cov:
            var_u = tf.matmul(SuTA, SuTA, transpose_a=True)
            var_v = (tf.matmul(SvTA, SvTA, transpose_a=True) + Cff -
                        tf.matmul(chol_KvvInv_Cvn, chol_KvvInv_Cvn, transpose_a=True))
        else:
            var_u = tf.reduce_sum(tf.square(SuTA), axis=0)
            var_v = (tf.reduce_sum(tf.square(SvTA), 0) + Cff -
                        tf.reduce_sum(tf.square(chol_KvvInv_Cvn), 0))

        return tf.squeeze(fu), tf.squeeze(fv), var_u, var_v

    # @tf.function
    def _build_expected_log_term(self, outputs, matrix_params):
        '''
        Computes E_q log p(y|f)
        @param data_inputs: input data
        @param data_targets: output data
        @return:
        '''

        fu, fv, var_u, var_v = self._build_mean_vars(matrix_params)
        means = fu + fv
        variances = var_u + var_v

        return self.likelihood.variational_expectations(means, variances, outputs)

    # @tf.function
    def _build_predictive_gaussian_mean_vars(self, matrix_params):
        '''
        Computes the mean and variance of E_q p(y|f)
        '''

        fu, fv, var_u, var_v = self._build_mean_vars(matrix_params)
        means = fu + fv
        variances = var_u + var_v

        return self.likelihood(means, variances)


    def get_joint_prediction(self, inputs):

        matrix_params = self.update_params(inputs, full_cov=True)

        fu, fv, var_u, var_v = self._build_mean_vars(matrix_params, full_cov=True)
        means = tf.reshape(fu + fv, (-1, 1))
        covariances = var_u + var_v
        

        return means, covariances