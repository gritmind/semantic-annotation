# This script is for ...
# converting from list to numpy array
# converting dummy nodes for output
# converting 2D Array for CNN1
# converting 3 time series set for CNN2, RNN

from data_handler import *
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



def dummy_to_integer(array):

    integer_array = [0] * len(array)

    for i, dummy in enumerate(array):
        for j, value in enumerate(dummy):
            if value == 1:
                integer_array[i] = j
                break

    return integer_array









def prepare_for_FNN(X_train, Y_train, X_test, Y_test):

	# make unfolded data (from sequence to non-sequence)
	unfolded_X_train, unfolded_Y_train = unfold_data(X_train, Y_train)
	unfolded_X_test, unfolded_Y_test = unfold_data(X_test, Y_test)

	#########################
	""" For training data """
	#########################
	# make numpy arrays
	unfolded_X_train = np.array(unfolded_X_train)
	unfolded_X_train = unfolded_X_train.astype(float)

	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(unfolded_Y_train)
	encoded_Y = encoder.transform(unfolded_Y_train)
	# convert integers to dummy variables (i.e. one hot encoded)
	unfolded_Y_train_d = np_utils.to_categorical(encoded_Y)

	#########################
	""" For testing data """
	#########################
	unfolded_X_test = np.array(unfolded_X_test)
	unfolded_X_test = unfolded_X_test.astype(float)
	# encode class values as integers
	encoder2 = LabelEncoder()
	encoder2.fit(unfolded_Y_test)
	encoded_Y2 = encoder2.transform(unfolded_Y_test)
	# convert integers to dummy variables (i.e. one hot encoded)
	unfolded_Y_test_d = np_utils.to_categorical(encoded_Y2)

	return unfolded_X_train, unfolded_Y_train_d, unfolded_X_test, unfolded_Y_test_d




def prepare_for_CNN(X_train, Y_train, X_test, Y_test):

	# make unfolded data
	unfolded_X_train, unfolded_Y_train_d, unfolded_X_test, unfolded_Y_test_d = prepare_for_FNN(X_train, Y_train, X_test, Y_test)


	#########################
	""" For training data """
	#########################
	# For make 2D array like image, add zero padding
	x_square = math.ceil(math.sqrt(len(unfolded_X_train[0]))) # square
	image_dim = x_square * x_square # e.g. 32 x 32 for image
	X_train_2D = [0] * len(unfolded_X_train)
	for i, point in enumerate(unfolded_X_train):
		X_train_2D[i] = np.lib.pad(unfolded_X_train[i], (0, image_dim - len(point)), 'constant', constant_values=(0, 0))
	#print('new array dim = %d' % len(X_train_2D[0]))
	#print('X_train_2D = (',len(X_train_2D),',', len(X_train_2D[0]),')')

	# Reshape
	for i, point in enumerate(X_train_2D):
		X_train_2D[i] = X_train_2D[i].reshape(int(x_square), int(x_square))
	# For making tensor input
	for i, point in enumerate(X_train_2D):
		X_train_2D[i] = [point]
	# make numpy arrays
	X_train_2D = np.array(X_train_2D)
	X_train_2D = X_train_2D.astype(float)
	#print('input tensor = ',X_train_2D.shape)
	#print('output dimension = %d' % unfolded_Y_train_d.shape[1])


	#########################
	""" For testing data """
	#########################
	# For make 2D array like image, add zero padding
	x_square2 = math.ceil(math.sqrt(len(unfolded_X_test[0]))) # square
	image_dim2 = x_square2 * x_square2 # e.g. 32 x 32 for image
	X_test_2D = [0] * len(unfolded_X_test)
	for i, point in enumerate(unfolded_X_test):
		X_test_2D[i] = np.lib.pad(unfolded_X_test[i], (0, image_dim2 - len(point)), 'constant', constant_values=(0, 0))

	# Reshape
	for i, point in enumerate(X_test_2D):
		X_test_2D[i] = X_test_2D[i].reshape(int(x_square2), int(x_square2))
	# For making tensor input
	for i, point in enumerate(X_test_2D):
		X_test_2D[i] = [point]
	# make numpy arrays
	X_test_2D = np.array(X_test_2D)
	X_test_2D = X_test_2D.astype(float)

	return X_train_2D, unfolded_Y_train_d, X_test_2D, unfolded_Y_test_d



#def prepare_for_CNN2(X_train, Y_train, X_test, Y_test):
