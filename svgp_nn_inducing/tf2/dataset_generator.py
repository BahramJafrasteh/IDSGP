from tensorflow import keras
import numpy as np
import math
import pandas as pd
import tensorflow as tf


class BigDatasetLoader():

    def __init__(self, folder_name, batch_size, shuffle = True, standardize_input = True, standardize_output = True, dtype=tf.float32):
        self.folder_name = folder_name
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.standardize_input = standardize_input
        self.standardize_output = standardize_output
        self.dtype = dtype
        
        info = np.array(pd.read_csv(folder_name + "/info.csv", header = None))
        self.n_train = info[0][0]
        self.n_test = info[0][1]
        self.n_attributes = info[0][2]
        self.type=info[0][3]

        if self.standardize_input:
            self.X_mean = np.array(pd.read_csv(self.folder_name + "/X_mean.csv", header = None))
            self.X_std = np.array(pd.read_csv(self.folder_name + "/X_std.csv", header = None))

        if self.standardize_output:
            self.y_mean = np.array(pd.read_csv(self.folder_name + "/y_mean.csv", header = None)).squeeze()
            self.y_std = np.array(pd.read_csv(self.folder_name + "/y_std.csv", header = None)).squeeze()
        else:
            self.y_mean = 0.0
            self.y_std = 1.0

        # Read train data
        self.input_train, self.output_train = self.read_data("train")
        self.input_test, self.output_test = self.read_data("test")

        self.dataset_train = tf.data.Dataset.zip((self.input_train, self.output_train)).batch(self.batch_size).repeat()
        self.dataset_test = tf.data.Dataset.zip((self.input_test, self.output_test)).batch(self.batch_size).repeat()

    def read_data(self, split="train"):
        """
        :param split train/test
        """

        if self.type == 'reg':
            out_type = self.dtype
        elif self.type == 'class':
            out_type = "int" + self.dtype.name[-2:]

        input = tf.data.experimental.CsvDataset(self.folder_name + "/" + split + "_data.csv", [self.dtype] * self.n_attributes, header=False)
        input = input.map(lambda *items: tf.stack(items))
        
        if self.standardize_input:
            input = input.map(lambda item: (item - self.X_mean.squeeze()) / self.X_std.squeeze())


        output = tf.data.experimental.CsvDataset(self.folder_name + "/" + split + "_labels.csv", [out_type], header=False)
        output = output.map(lambda *items: tf.stack(items))      

        if self.standardize_output:
            output = output.map(lambda item: (item - self.y_mean) / self.y_std) 

        return input, output

    def get_train(self):

        return self.dataset_train

    
    def get_test(self):

        return self.dataset_test

    def estimate_lengthscale(self):
        X_sample = self.sample(n_samples=1000)

        dist2 = np.sum(X_sample**2, 1, keepdims = True) - 2.0 * np.dot(X_sample, X_sample.T) + np.sum(X_sample**2, 1, keepdims = True).T
        log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(1000, 1) ]))

        return log_l

    def sample(self, n_samples=100):
        
        X_sample = np.array([s for s in self.input_train.take(n_samples).as_numpy_iterator()]).squeeze()

        return X_sample

