'''
Created on 13 mar. 2017

@author: carlos
'''

import tensorflow as tf
from svgp_nn_inducing.tf2.kernel import Kernel
import svgp_nn_inducing.tf2.settings as settings
        
class RBF_ARD(Kernel):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    
    def __init__(self, log_lengthscales, log_sigma0, log_sigma, jitter = 1e-3):
            
        super(RBF_ARD, self).__init__(jitter)
        
        self.log_lengthscales = tf.Variable(log_lengthscales, dtype = settings.tf_float_type)
        self.log_sigma0 = tf.Variable([ log_sigma0 ], dtype = settings.tf_float_type)
        self.log_sigma = tf.Variable([ log_sigma ], dtype = settings.tf_float_type)
        self.jitter = tf.constant([ jitter ], dtype = settings.tf_float_type)

    @tf.function
    def call(self, X, X2 = None):

        """
        This function computes the covariance matrix for the GP
        """
        if X2 is None:
            X2 = X
            if (len(tf.shape(X)) == 2):
                eye_matrix = tf.eye(tf.shape(input=X)[ 0 ], dtype = settings.tf_float_type)
            else:
                eye_matrix = tf.eye(tf.shape(input=X)[ 1 ], batch_shape = [tf.shape(input=X)[ 0 ]], dtype = settings.tf_float_type)
            white_noise = (self.jitter + tf.exp(self.log_sigma0)) * eye_matrix
        else:
            white_noise = 0.0
            
        if len(tf.shape(X)) > len(tf.shape(X2)):
            X2 = tf.expand_dims(X2, 1)
        elif len(tf.shape(X2)) > len(tf.shape(X)):
            X = tf.expand_dims(X, 1)
            
        X = X / tf.sqrt(tf.exp(self.log_lengthscales))
        X2 = X2 / tf.sqrt(tf.exp(self.log_lengthscales))
  
        distance = self._compute_distance(X, X2)
        
        return tf.exp(self.log_sigma) * tf.exp(-0.5 * distance) + white_noise
        
    @tf.function
    def _compute_distance(self, X1, X2):
        """
        Computes pairwise Euclidean distances between x1 and x2
        Can compute batch distances when dimensionality = 3
        Args:
          x1,    [?,m,d] matrix
          x2,    [?,n,d] matrix
        Returns:
          covar,    [?,m,n] Euclidean distances
        """
        
        value = tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(X1), axis=-1), -1)
        
        if len(tf.shape(X2)) == 2:
            value2 = tf.transpose(a=tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(X2), axis=-1), -1))
        else:
            value2 = tf.transpose(a=tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(X2), axis=-1), -1), perm=[0, 2, 1])
        
        covar = tf.maximum(value - 2 * tf.matmul(X1, X2, False, True) + value2, 0.0)
        
        covar = tf.sqrt(tf.maximum(covar, 1e-40))
        
        return covar

    def get_params(self):
        return [ self.log_lengthscales, self.log_sigma, self.log_sigma0 ]

    def get_log_sigma(self):
        return self.log_sigma

    def get_log_sigma0(self):
        return self.log_sigma0

    @tf.function
    def get_var_points(self, data_points):
        return  tf.ones([ tf.shape(input=data_points)[ 0 ] ], dtype = settings.tf_float_type) * tf.exp(self.log_sigma) + (self.jitter + tf.exp(self.log_sigma0))

 

