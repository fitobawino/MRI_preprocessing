# Flores-Saiffe Farías Adolfo
# Universidad de Guadalajara
# adolfo.flores.saiffe@gmail.com
# saiffe.adolfo@alumnos.udg.mx
# ORCID:0000-0003-4504-7251
# Researchgate: https://www.researchgate.net/profile/Adolfo_Flores_Saiffe2
# Github: https://github.com/fitobawino
#
# Utilery for ANN_fMRIs.py
#

import os, random, itertools
import matplotlib.pyplot as plt
import numpy as np
import h5py
from os.path import join as opj
from keras import regularizers
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, ActivityRegularization
from keras.callbacks import TensorBoard, EarlyStopping
from time import time

def export_to_h5(X, Y, path):
# Function to save data to path
	import h5py
	file = h5py.File(path + "_db.hdf5", "w")
	file.create_dataset('X_data', data=X)
	file.create_dataset('Y_data', data=Y)
	file.close()

def data_augmentation(X, Y, splits):
# Funtion to split data augmentation in split parts
	a,b,c = X.shape
	X2 = np.zeros([int(a*splits), b, int(c/splits)], dtype=float)
	Y2 = np.zeros([a*splits, 2], dtype=float)
	line = 0
	for x in range(0, len(X)):
		aux = np.split(X[x], splits, axis=1)
		for split in range(0, splits):
			X2[line] = aux[split]
			Y2[line] = Y[x]
			line += 1
	return X2, Y2

def split_data(X, Y, test_size):
# Split data into train and test
	train = np.array(random.sample(range(0,len(X)), test_size))
	train = np.append(train, train + 16)
	x_train = np.delete(X, train, 0)
	y_train = np.delete(Y, train, 0)
	x_test = np.zeros([test_size * 2, 366*time_length])
	y_test = np.zeros([test_size * 2, 2])
	for i in range(0,len(train)):
		x_test[i] = X[train[i],:][:]
		y_test[i] = Y[train[i],:][:]
	return x_test, y_test, x_train, y_train

def baseline_model(LSTM_size, time_len, ROIs, drop, L2_reg):
# Baseline model. Layers: LSTM + Dense + Dense (output)
	model = Sequential()
	model.add(LSTM(LSTM_size, # input (Samples, Time, Features)
		return_sequences=True, # one output for each input time step
		activation='relu',
		kernel_regularizer=regularizers.l1_l2(l1=L2_reg, l2=L2_reg),
		dropout=drop,
		input_shape=(time_len, ROIs))) #128/567
	model.add(Flatten())
	model.add(Dense(LSTM_size, activation='sigmoid'))
	model.add(Dense(2, activation='softmax'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) #binary_crossentropy
	return model

def model_bidirectional(LSTM_size, time_len, ROIs, drop, L2_reg):
# Same as baseline but bidirectional LSTM instead of regular LSTM
	model = Sequential()
	model.add(Bidirectional(LSTM(LSTM_size, 
		return_sequences=True, # one output for each input time step
		activation='relu',
		kernel_regularizer=regularizers.l1_l2(l1=L2_reg, l2=L2_reg),
		dropout=drop),
		input_shape=(time_len, ROIs))) #128/567
	model.add(Flatten())
	model.add(Dense(LSTM_size, activation='sigmoid'))
	model.add(Dense(2, activation='softmax'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	return model

def model_deepLSTM(LSTM_size, time_len, ROIs, drop, L2_reg):
# Same as baseline but adding 2 additional layers of LSTM.
	model = Sequential()
	for n in range(0, 3):
		model.add(LSTM(int(LSTM_size), # input (Samples, Time, Features)
			return_sequences=True, # one output for each input time step
			activation='relu',
			kernel_regularizer=regularizers.l1_l2(l1=L2_reg, l2=L2_reg),
			dropout=drop,
			input_shape=(time_len, ROIs))) #128/567
	model.add(Flatten())
	model.add(Dense(2, activation='softmax'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) #binary_crossentropy
	return model

def model_deepdense(LSTM_size, time_len, ROIs, drop, L2_reg):
#Same as baseline but adding 2 layers of dense.
	model = Sequential()
	model.add(LSTM(LSTM_size, 
		return_sequences=True, # one output for each input time step
		activation='relu',
		kernel_regularizer=regularizers.l1_l2(l1=L2_reg, l2=L2_reg),
		dropout=drop,
		input_shape=(time_len, ROIs))) #128/567
	model.add(Flatten())
	for n in range(0, 3):
		model.add(Dense(LSTM_size, 
		activation='sigmoid'))
		model.add(Dropout(drop)) ## Dense networks with LSTM output size
	model.add(Dense(2, activation='softmax')) 
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
	return model

def model_nodense(LSTM_size, time_len, ROIs, drop, L2_reg):
#Same as baseline but removing the dense hidden layer.
	model = Sequential()
	model.add(LSTM(int(LSTM_size), # input (Samples, Time, Features)
		return_sequences=True, # one output for each input time step
		activation='relu',
		kernel_regularizer=regularizers.l1_l2(l1=L2_reg, l2=L2_reg),
		dropout=drop,
		input_shape=(time_len, ROIs))) #128/567
	model.add(Flatten())
	model.add(Dense(2, activation='softmax'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) #binary_crossentropy
	return model

def save_and_plot(hist, it, modelname, graph, path):
# Save and plot results
	if graph:
		plt.plot(hist['acc'])
		plt.plot(hist['val_acc'])
		plt.title('model')
		plt.ylabel('acc/val_acc')
		plt.xlabel('epoch')
		plt.legend(['Training', 'Validation'], loc='upper left')
		plt.savefig(opj(path,  modelname) + "plot" + it + ".pdf")
		plt.close()
	#save vars
	np.savetxt(opj(path,  modelname) + "_loss_" + it, hist['loss'], delimiter="\t")
	np.savetxt(opj(path,  modelname) + modelname + "_acc_" + it, hist['acc'], delimiter="\t")
	
def import_from_h5(path):
# Import X, Y data from a h5
	file = h5py.File(path,'r')
	X = file['X_data'][:]
	Y = file['Y_data'][:]
	return X, Y
