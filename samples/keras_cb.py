import keras

from math import log, sqrt
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE


class TestAccuracyCallback(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.accs = []

    def on_epoch_end(self, epoch, logs={}):
        cur_acc = logs.get('val_acc')
        self.accs.append(cur_acc)
        cur_loss = 1.0 - cur_acc
        elapsed_time = time.time() - self.start_time
        max_i = np.argmax(self.accs)
        debug("Training {} epoch(s) takes {:.1f} secs. Current best test accuracy {} is at epoch {}".format(
            epoch, elapsed_time, cur_acc, self.accs[max_i], max_i+1))
        update_result_per_epoch(epoch, cur_loss, elapsed_time)


class RMSELossCallback(keras.callbacks.Callback):
    
	def on_train_begin(self, logs={}):
		self.start_time = time.time()
		self.losses = []

	def on_epoch_end(self, i, logs={}):
    		#print(self.validation_data[0])
		p = self.model.predict(self.validation_data[0], batch_size=self.params['batch_size'])
		mse = MSE(self.validation_data[1], p)
		rmse = sqrt(mse)        
		self.losses.append(rmse)
		elapsed_time = time.time() - self.start_time
		print("Training {} epoches takes {:.1f} secs.\n validation losses: {}".format(
			i+1, elapsed_time, self.losses))