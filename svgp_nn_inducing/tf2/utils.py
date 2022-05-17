import tensorflow as tf
import numpy as np
import os
def get_num_params(params, n_par = 0):
    """
    @param params: trainable parameters
    @param n_par: number of parameters
    @return: final number of trainable parameters
    """

    for param in params:
        if isinstance(param, list):
            n_par = get_num_params(param, n_par)
        else:
            n_par += np.prod(param.get_shape().as_list())
            
    return int(n_par)

def generate_layer_name(name, prefix=None, layer_id = 0):
    return '_'.join((prefix, 'layer', str(layer_id), name))
    




################################# Bernoulli ###############################################

def cdf_normal(x):
    # CDF = 0.5 * ( 1 + erf(x/ sqrt(2.0) ) )
    epsilon = 1e-3 
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * epsilon) + epsilon




# save models to the disk
def save_model(model, optimizer, model_path, epoch = 'latest'):
    save_filename = 'VSGP_%s.pckt' % (epoch)
    save_path = os.path.join(model_path, 'checkpoints/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_filepath = os.path.join(model_path, 'checkpoints/', save_filename)
    #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #checkpoint.save(file_prefix=save_filepath)
    model.save_weights(save_filepath)
    print ('Model successfully save to %s' % save_filepath)

def load_model(model, optimizer, model_path, which_epoch =  'latest'):
    # load models from the disk
    load_filename = 'VSGP_%s.pckt' % (which_epoch)

    load_path = os.path.join(model_path, 'checkpoints/', load_filename)


    print('loading the model from %s' % load_path)
    #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    model.load_weights(load_path).expect_partial()
    #status = checkpoint.restore(tf.train.latest_checkpoint(load_path))
    #if status:
    print ('Model successfully loaded from %s' % load_path)






