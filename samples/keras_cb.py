import keras
import time
import numpy as np
from math import log, sqrt
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from ws.apis import update_loss_per_epoch
from ws.shared.logger import *

class TestAccuracyCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.accs = []

    def on_epoch_end(self, i, logs={}):
        cur_acc = logs.get('val_acc')
        num_epoch = i + 1
        self.accs.append(cur_acc)
        cur_loss = 1.0 - cur_acc
        elapsed_time = time.time() - self.start_time
        max_i = np.argmax(self.accs)
        debug("Training {} epoch(s) yields {}({:.1f} secs). Current best accuracy {} is at epoch {}.".format(
              num_epoch, cur_acc, elapsed_time, self.accs[max_i], max_i+1))
        # XXX:update result via file
        update_current_loss(num_epoch, cur_loss, elapsed_time)


class RMSELossCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.losses = []

    def on_epoch_end(self, i, logs={}):
        num_epoch = i + 1
        p = self.model.predict(self.validation_data[0],
                               batch_size=self.params['batch_size'])
        mse = MSE(self.validation_data[1], p)
        rmse = sqrt(mse)
        self.losses.append(rmse)
        elapsed_time = time.time() - self.start_time
        # XXX:update result via file
        update_current_loss(num_epoch, rmse, elapsed_time, loss_type='rmse')
        debug("Training {} epoches takes {:.1f} secs. Current min loss: {}".format(
            num_epoch, elapsed_time, min(self.losses)))
