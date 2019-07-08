import time
import os

try:
	import keras
	from keras.datasets import mnist
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers.normalization import BatchNormalization as BatchNorm
	from keras.callbacks import EarlyStopping
	from keras import backend as K
except:
	raise ImportError("For this example you need to install keras.")

try:
	import torchvision
	import torchvision.transforms as transforms
except:
	raise ImportError("For this example you need to install pytorch-vision.")

import os
import logging
logging.basicConfig(level=logging.DEBUG)

#from common_defs import *
from math import log, sqrt
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from load_data_for_regression import data
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler


def print_layers(params):
    for i in range(1, params['n_layers'] + 1):
        print("layer {} | size: {:>3} | activation: {:<7} | extras: {}".format(i,
                    params['layer_{}_size'.format(i)],
                    params['layer_{}_activation'.format(i)],
                    params['layer_{}_extras'.format(i)]['name'])
        )
    if params['layer_{}_extras'.format(i)]['name'] == 'dropout':
        print("- rate: {:.1%}".format(params['layer_{}_extras'.format(i)]['rate']), )


def print_params(params):
	print({k: v for k, v in params.items() if not k.startswith('layer_')})
	print_layers(params)

    
def make_dict_layer_extras(param, layer_extra_name):
	layer_number = layer_extra_name.split('_')[1]
	if param[layer_extra_name] == 'None':
		param[layer_extra_name] = {'name': 'None'}
	elif param[layer_extra_name] == 'batchnorm':
		param[layer_extra_name] = {'name': 'batchnorm'}
	else:
		param[layer_extra_name] = {'name': 'dropout', 'rate': param['dropout_rate_{}'.format(layer_number)]}

        
def config_to_params(input_config):
	params = input_config

	if 'layer_1_extras' in params:
		make_dict_layer_extras(params, 'layer_1_extras')
	if 'layer_2_extras' in params:
		make_dict_layer_extras(params, 'layer_2_extras')
	if 'layer_3_extras' in params:
		make_dict_layer_extras(params, 'layer_3_extras')
	if 'layer_4_extras' in params:
		make_dict_layer_extras(params, 'layer_4_extras')
	if 'layer_5_extras' in params:
		make_dict_layer_extras(params, 'layer_5_extras')

	try:
		params.pop('dropout_rate_1', None)
		params.pop('dropout_rate_2', None)
		params.pop('dropout_rate_3', None)
		params.pop('dropout_rate_4', None)
		params.pop('dropout_rate_5', None)
	except:
		print()
	return(params)


class KerasWorker(object):
	def __init__(self, N_train=8192, N_valid=1024, **kwargs):
		#super().__init__(**kwargs)
		pass

	def compute(self, config, budget, working_directory, history, *args, **kwargs):
		"""
		Simple example for a compute function using a feed forward network.
		It is trained on the MNIST dataset.
		The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
		"""
		params = config_to_params(config)
		n_iterations = budget

		print("Total iterations:", n_iterations)

		y_train = data['y_train']
		y_test = data['y_test']

		if params['scaler'] != 'None':
			scaler = eval("{}()".format(params['scaler']))
			x_train_ = scaler.fit_transform(data['x_train'].astype(float))
			x_test_ = scaler.transform(data['x_test'].astype(float))
		else:
			x_train_ = data['x_train']
			x_test_ = data['x_test']

		input_dim = x_train_.shape[1]

		model = Sequential()
		model.add(Dense(params['layer_1_size'], init=params['init'],
						activation=params['layer_1_activation'], input_dim=input_dim))

		for i in range(int(params['n_layers']) - 1):

			extras = 'layer_{}_extras'.format(i + 1)

			if params[extras]['name'] == 'dropout':
				model.add(Dropout(params[extras]['rate']))
			elif params[extras]['name'] == 'batchnorm':
				model.add(BatchNorm())

			model.add(Dense(params['layer_{}_size'.format(i + 2)], init=params['init'],
							activation=params['layer_{}_activation'.format(i + 2)]))

		model.add(Dense(1, init=params['init'], activation='linear'))
		model.compile(optimizer=params['optimizer'], loss=params['loss'])


		validation_data = (x_test_, y_test)
		early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
		h = model.fit(x_train_, y_train,
							epochs=int(round(n_iterations)),
							batch_size=params['batch_size'],
							shuffle=params['shuffle'],
							validation_data=validation_data,
							callbacks=[early_stopping, history])

		#p = model.predict(x_train_, batch_size=params['batch_size'])
		#mse = MSE(y_train, p)
		#rmse = sqrt(mse)
		#mae = MAE(y_train, p)
		#print("\n# training | RMSE: {:.4f}, MAE: {:.4f}".format(rmse, mae))

		p = model.predict(x_test_, batch_size=params['batch_size'])

		mse = MSE(y_test, p)
		rmse = sqrt(mse)
		mae = MAE(y_test, p)

		print("# testing  | RMSE: {:.4f}, MAE: {:.4f}".format(rmse, mae))
		return ({'cur_loss': rmse,
				 'loss_type': 'rmse',
				 'cur_iter' : len(h.history['loss']),
				 'iter_unit': 'epoch',                 
				 'early_stop': model.stop_training, 
				 'info' : {
					'params': params,
					'mae': mae}})

	@staticmethod
	def get_configspace():

		import ConfigSpace as CS
		import ConfigSpace.hyperparameters as CSH    		

		"""
		It builds the configuration space with the needed hyperparameters.
		It is easily possible to implement different types of hyperparameters.
		Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
		:return: ConfigurationsSpace-Object
		"""
		cs = CS.ConfigurationSpace()

		scaler = CSH.CategoricalHyperparameter('scaler', ['None', 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler'])
		init = CSH.CategoricalHyperparameter('init', ['uniform', 'normal', 'glorot_uniform',
		'glorot_normal', 'he_uniform', 'he_normal' ])
		batch_size = CSH.CategoricalHyperparameter('batch_size', [16, 32, 64, 128, 256 ])
		shuffle = CSH.CategoricalHyperparameter('shuffle', [True, False])
		loss = CSH.CategoricalHyperparameter('loss', ['mean_absolute_error', 'mean_squared_error' ])
		optimizer = CSH.CategoricalHyperparameter('optimizer', ['rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax'])

		cs.add_hyperparameters([scaler, init, batch_size, shuffle, loss, optimizer])

		n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=5, default_value=2)
        

		layer_sizes = [
			CSH.UniformIntegerHyperparameter('layer_{}_size'.format(l), lower=2, upper=100, default_value=16, log=True)
			for l in range(1, 6)
		]


		layer_activations = [
			CSH.CategoricalHyperparameter('layer_{}_activation'.format(l), ['relu', 'sigmoid', 'tanh'])
			for l in range(1, 6)
		]            

		layer_extras = [
			CSH.CategoricalHyperparameter('layer_{}_extras'.format(l), ['None', 'dropout', 'batchnorm'])
			for l in range(1, 6)
		]            

		dropout_rates = [
			CSH.UniformFloatHyperparameter('dropout_rate_{}'.format(l), lower=0.1, upper=0.5, default_value=0.2, log=False)
			for l in range(1, 6)
		]

		cs.add_hyperparameters([n_layers] + layer_sizes + layer_activations + layer_extras + dropout_rates)  
        
		conditions = [ CS.GreaterThanCondition(layer_sizes[n], n_layers, n) for n in range(1, 5)]        

		conditions = conditions + [ CS.GreaterThanCondition(layer_activations[n], n_layers, n) for n in range(1, 5) ]           

		conditions = conditions + [ CS.GreaterThanCondition(layer_extras[n], n_layers, n) for n in range(1, 5) ]          

		equal_conditions = [ CS.EqualsCondition(dropout_rates[n], layer_extras[n], 'dropout') for n in range(0, 5) ]  

		greater_size_conditions = [ CS.GreaterThanCondition(dropout_rates[n], n_layers, n) for n in range(1, 5) ]          


		for c in conditions:
			cs.add_condition(c)            

		cs.add_condition(equal_conditions[0])
     
        
		for j in range(0, 4):
			cond = CS.AndConjunction(greater_size_conditions[j], equal_conditions[j+1])
			cs.add_condition(cond)         

		return cs


if __name__ == "__main__":
    from samples.keras_cb import RMSELossCallback
    start_time = time.time()
    gpu_id = 0
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)    
    
    worker = KerasWorker(run_id=str(gpu_id))
    cs = worker.get_configspace()
    history = RMSELossCallback()
    config = cs.sample_configuration().get_dictionary()
    
    res = worker.compute(config=config, budget=27, history=history, working_directory='.', params=config)
    elapsed = time.time() - start_time

    #print_params(res['info']['params'])
    print("Result: {}".format(res))
    print("Elapsed time: {}, Final RMSE: {:.2%}".format(elapsed, res['cur_loss']))
    