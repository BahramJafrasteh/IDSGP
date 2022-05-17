import tensorflow as tf
import time
from svgp_nn_inducing.tf2.utils import save_model

class EpochCSVLogger(tf.keras.callbacks.CSVLogger):

    def __init__(self, data_train, data_test, batch_size = None, steps_train = None, steps_test = None, filename = "./training.log", separator=',', append=False, predict_test=True):

        if isinstance(data_train, tuple):
            self.X_train, self.y_train = data_train
            self.data_train = None
        else:
            self.data_train = data_train

        if isinstance(data_test, tuple):
            self.X_test, self.y_test = data_test
            self.data_test = None
        else:
            self.data_test = data_test

        self.batch_size = batch_size
        self.filename_ = filename
        self.predict_test = predict_test
        self.steps_train = steps_train
        self.steps_test = steps_test
        super(EpochCSVLogger, self).__init__(filename, separator=',', append=False)

    def on_train_begin(self, logs=None):
        self.total_time_training = 0.0
        self.predict_time = 0
        super(EpochCSVLogger, self).on_train_begin()

    def on_train_end(self, logs=None):
        if self.data_train is not None:
            _, rmse_train, nll_train = self.model.evaluate(self.data_train, batch_size = self.batch_size, steps=self.steps_train)
        else:
            _, rmse_train, nll_train = self.model.evaluate(self.X_train, self.y_train, batch_size = self.batch_size, steps=self.steps_train)
        start_predict = time.time()
        if self.data_test is not None:
            _, rmse_test, nll_test = self.model.evaluate(self.data_test, batch_size = self.batch_size, steps=self.steps_test)
        else:
            _, rmse_test, nll_test = self.model.evaluate(self.X_test, self.y_test, batch_size = self.batch_size, steps=self.steps_test)
        end_predict = time.time()
        self.predict_time = end_predict - start_predict
        print("RMSE_train {}, NLL_train {}, RMSE_test {}, NLL_test {}, total_training_time {}, prediction_time {}".format(
            rmse_train, nll_train, rmse_test, nll_test, self.total_time_training, self.predict_time) )
        filename_extension = self.filename.split(".")
        with open(filename_extension[0] + '_final.'+ filename_extension[1], "w") as myfile:
            myfile.write('RMSE_train, NLL_train, RMSE_test, NLL_test, total_training_time, prediction_time' '\n')
            myfile.write(str(rmse_train) + " " + str(nll_train) + " " + str(rmse_test) +
                         " " + str(nll_test) + " " + str(self.total_time_training) + " " + str(self.predict_time) + '\n')

        super(EpochCSVLogger, self).on_train_end()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        training_time = time.time() - self.start_epoch
        self.total_time_training += training_time
        logs['total_time_train'] = training_time
        if self.predict_test:
            if self.data_test is not None:
                _, err_test, nll_test = self.model.evaluate(self.data_test, batch_size = self.batch_size, steps = self.steps_test)
            else:
                _, err_test, nll_test = self.model.evaluate(self.X_test, self.y_test, batch_size = self.batch_size, steps = self.steps_test)
            logs[self.model.metrics_names[1] + '_test'] = err_test
            logs['nll_test'] = nll_test
        super(EpochCSVLogger, self).on_epoch_end(epoch, logs)


class TimerStopper(tf.keras.callbacks.Callback):
    def __init__(self, data_train, data_test, batch_size = None, batch_size_test = None, steps_train = None, steps_test = None, filename = "./training.log", max_seconds = 60, path_results = "model/"):
        if isinstance(data_train, tuple):
            self.X_train, self.y_train = data_train
            self.data_train = None
        else:
            self.data_train = data_train

        if isinstance(data_test, tuple):
            self.X_test, self.y_test = data_test
            self.data_test = None
        else:
            self.data_test = data_test

        self.max_seconds = max_seconds
        self.path_results = path_results
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.steps_train = steps_train
        self.steps_test = steps_test
        self.filename = filename

        super(TimerStopper, self).__init__()

    def _save_results(self):
        if self.data_train is not None:
            _, rmse_train, nll_train = self.model.evaluate(self.data_train, batch_size = self.batch_size, steps=self.steps_train)
        else:
            _, rmse_train, nll_train = self.model.evaluate(self.X_train, self.y_train, batch_size = self.batch_size, steps=self.steps_train)
        start_predict = time.time()
        if self.data_test is not None:
            _, rmse_test, nll_test = self.model.evaluate(self.data_test, batch_size = self.batch_size_test, steps=self.steps_test)
        else:
            _, rmse_test, nll_test = self.model.evaluate(self.X_test, self.y_test, batch_size = self.batch_size_test, steps=self.steps_test)
        end_predict = time.time()
        self.predict_time = end_predict - start_predict
        print("RMSE_train {}, NLL_train {}, RMSE_test {}, NLL_test {}, total_training_time {}, prediction_time {}".format(
            rmse_train, nll_train, rmse_test, nll_test, self.total_time_training, self.predict_time) )
        filename_extension = self.filename.split(".")
        with open(filename_extension[0] + '_final.'+ filename_extension[1], "w") as myfile:
            myfile.write('RMSE_train, NLL_train, RMSE_test, NLL_test, total_training_time, prediction_time' '\n')
            myfile.write(str(rmse_train) + " " + str(nll_train) + " " + str(rmse_test) +
                         " " + str(nll_test) + " " + str(self.total_time_training) + " " + str(self.predict_time) + '\n')

    def on_train_begin(self, logs=None):
        self.total_time_training = 0.0
       
        super(TimerStopper, self).on_train_begin(logs=None)

    def on_train_end(self, logs=None):
        save_model(self.model, None, self.path_results)

        self._save_results()

        super(TimerStopper, self).on_train_end(logs=None)

    def on_train_batch_begin(self, batch, logs=None):
        self.start_epoch_batch = time.time()

        super(TimerStopper, self).on_train_batch_begin(batch, logs=None)

    def on_train_batch_end(self, batch, logs=None):
        training_time = time.time() - self.start_epoch_batch
        self.total_time_training += training_time

        if  self.total_time_training > self.max_seconds:
            self.model.stop_training = True
            

class NBatchCSVLogger(tf.keras.callbacks.CSVLogger):

    def __init__(self, data_test, batch_size = None, steps_test = None, filename = "./training.log", separator=',', append=False, each_n_batches=500):
        self.data_test = data_test
        self.batch_size = batch_size
        self.filename_ = filename
        self.filename_batch = filename[:filename.rfind('.txt')] + '_batch'+'.txt'
        self.each_n_batches = each_n_batches
        self.steps_test = steps_test
        super(NBatchCSVLogger, self).__init__(filename, separator=',',append=False)

    def on_train_begin(self, logs=None):
        self.total_time_training = 0.0
        self.predict_time = 0
        with open(self.filename_batch, "w") as myfile:
            myfile.write("err_test" + " " + "nll_test" + " " +
                            "training_time" +'\n')
        super(NBatchCSVLogger, self).on_train_begin()

    def on_train_end(self, logs=None):
        # _, rmse_train, nll_train = self.model.evaluate(self.X_train, self.y_train, batch_size = self.batch_size)
        start_predict = time.time()
        _, rmse_test, nll_test = self.model.evaluate(self.data_test, batch_size = self.batch_size, steps = self.steps_test)
        end_predict = time.time()
        self.predict_time = end_predict - start_predict
        # print("RMSE_train {}, NLL_train {}, RMSE_test {}, NLL_test {}, total_training_time {}, prediction_time {}".format(
        #     rmse_train, nll_train, rmse_test, nll_test, self.total_time_training, self.predict_time) )
        # filename_extension = self.filename.split(".")
        # with open(filename_extension[0] + '_final'+ filename_extension[1], "w") as myfile:
        #     myfile.write('RMSE_train, NLL_train, RMSE_test, NLL_test, total_training_time, prediction_time' '\n')
        #     myfile.write(str(rmse_train) + " " + str(nll_train) + " " + str(rmse_test) +
        #                  " " + str(nll_test) + " " + str(self.total_time_training) + " " + str(self.predict_time) + '\n')
        print("RMSE_test {}, NLL_test {}, total_training_time {}, prediction_time {}".format(
            rmse_test, nll_test, self.total_time_training, self.predict_time) )
        filename_extension = self.filename.split(".")
        with open(filename_extension[0] + '_final.'+ filename_extension[1], "w") as myfile:
            myfile.write('RMSE_test, NLL_test, total_training_time, prediction_time' '\n')
            myfile.write(str(rmse_test) +
                         " " + str(nll_test) + " " + str(self.total_time_training) + " " + str(self.predict_time) + '\n')

        super(NBatchCSVLogger, self).on_train_end()

    def on_train_batch_begin(self, batch, logs=None):
        self.start_epoch_batch = time.time()

        super(NBatchCSVLogger, self).on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        training_time = time.time() - self.start_epoch_batch
        self.total_time_training += training_time

        if batch % self.each_n_batches == 0:
            _, err_test, nll_test = self.model.evaluate(self.data_test, batch_size = self.batch_size, steps = self.steps_test)
            
            with open(self.filename_batch, "a") as myfile:
                myfile.write(str(err_test) + " " + str(nll_test) + " " +
                            str(self.total_time_training) +'\n')

        
