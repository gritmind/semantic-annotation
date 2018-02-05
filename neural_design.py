from data_handler import *
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


def feedforward(input_dim_FNN, hidden_dim_FNN, output_dim_FNN):
    model = Sequential()
    model.add(Dense(hidden_dim_FNN, input_dim = input_dim_FNN, init='normal', activation='relu'))
    model.add(Dense(output_dim_FNN, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def convolutional(x_square, output_dim):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, x_square, x_square), activation='relu', border_mode='same'))
#     model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#     model.add(Dropout(0.2))
#     model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = 0.1
    decay = lrate/n_epoch_cnn
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())
    return model

def recurrent(feature_dim, hidden_dim_RNN, output_dim, time_length):
    model = Sequential()
    model.add(LSTM(hidden_dim_RNN, input_shape=(time_length, feature_dim)))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    return model

def dummy_to_integer(array):
    integer_array = [0] * len(array)
    for i, dummy in enumerate(array):
        for j, value in enumerate(dummy):
            if value == 1:
                integer_array[i] = j
                break
    return integer_array

def prepare_for_RNN(X_train, Y_train, X_test, Y_test):
    zero_array = [0]*len(X_train[0][0])
    padding_X_train = []
    for i, sentence in enumerate(X_train):
        for j, token in enumerate(sentence):
            temp_point = []
            if j==0 or j==len(sentence)-1:
                temp_point.append(zero_array)
                temp_point.append(zero_array)
                temp_point.append(sentence[j])
                padding_X_train.append(temp_point)
            elif j==1 or j==len(sentence)-2:
                temp_point.append(zero_array)
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_train.append(temp_point)
            else:
                temp_point.append(sentence[j-2])
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_train.append(temp_point)
    padding_X_train = np.array(padding_X_train, dtype='float32')
    unfolded_Y_train = []
    for sentence in Y_train:
        for token in sentence:
            unfolded_Y_train.append(token)
    encoder = LabelEncoder()
    encoder.fit(unfolded_Y_train)
    encoded_Y = encoder.transform(unfolded_Y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)
    ###############################################################
    zero_array = [0]*len(X_test[0][0])
    padding_X_test = []
    for i, sentence in enumerate(X_test):
        for j, token in enumerate(sentence):
            temp_point = []
            if j==0 or j==len(sentence)-1:
                temp_point.append(zero_array)
                temp_point.append(zero_array)
                temp_point.append(sentence[j])
                padding_X_test.append(temp_point)
            elif j==1 or j==len(sentence)-2:
                temp_point.append(zero_array)
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_test.append(temp_point)
            else:
                temp_point.append(sentence[j-2])
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_test.append(temp_point)
    padding_X_test = np.array(padding_X_test, dtype='float32')
    unfolded_Y_test = []
    for sentence in Y_test:
        for token in sentence:
            unfolded_Y_test.append(token)
    encoder = LabelEncoder()
    encoder.fit(unfolded_Y_test)
    encoded_Y = encoder.transform(unfolded_Y_test)
    dummy_y_test = np_utils.to_categorical(encoded_Y)
    return padding_X_train, dummy_y, padding_X_test, dummy_y_test

def prepare_for_FNN(X_train, Y_train, X_test, Y_test):
	# make unfolded data (from sequence to non-sequence)
	unfolded_X_train, unfolded_Y_train = unfold_data(X_train, Y_train)
	unfolded_X_test, unfolded_Y_test = unfold_data(X_test, Y_test)
	### train set
	# make numpy arrays
	unfolded_X_train = np.array(unfolded_X_train)
	unfolded_X_train = unfolded_X_train.astype(float)
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(unfolded_Y_train)
	encoded_Y = encoder.transform(unfolded_Y_train)
	# convert integers to dummy variables (i.e. one hot encoded)
	unfolded_Y_train_d = np_utils.to_categorical(encoded_Y)
	### test set
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
	### train set
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
	### test set
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

def prepare_for_RNN(X_train, Y_train, X_test, Y_test):
    zero_array = [0]*len(X_train[0][0])
    padding_X_train = []
    for i, sentence in enumerate(X_train):
        for j, token in enumerate(sentence):
            temp_point = []
            if j==0 or j==len(sentence)-1:
                temp_point.append(zero_array)
                temp_point.append(zero_array)
                temp_point.append(sentence[j])
                padding_X_train.append(temp_point)
            elif j==1 or j==len(sentence)-2:
                temp_point.append(zero_array)
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_train.append(temp_point)
            else:
                temp_point.append(sentence[j-2])
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_train.append(temp_point)
    padding_X_train = np.array(padding_X_train, dtype='float32')
    unfolded_Y_train = []
    for sentence in Y_train:
        for token in sentence:
            unfolded_Y_train.append(token)
    encoder = LabelEncoder()
    encoder.fit(unfolded_Y_train)
    encoded_Y = encoder.transform(unfolded_Y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)
    ###############################################################
    zero_array = [0]*len(X_test[0][0])
    padding_X_test = []
    for i, sentence in enumerate(X_test):
        for j, token in enumerate(sentence):
            temp_point = []
            if j==0 or j==len(sentence)-1:
                temp_point.append(zero_array)
                temp_point.append(zero_array)
                temp_point.append(sentence[j])
                padding_X_test.append(temp_point)
            elif j==1 or j==len(sentence)-2:
                temp_point.append(zero_array)
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_test.append(temp_point)
            else:
                temp_point.append(sentence[j-2])
                temp_point.append(sentence[j-1])
                temp_point.append(sentence[j])
                padding_X_test.append(temp_point)
    padding_X_test = np.array(padding_X_test, dtype='float32')
    unfolded_Y_test = []
    for sentence in Y_test:
        for token in sentence:
            unfolded_Y_test.append(token)
    encoder = LabelEncoder()
    encoder.fit(unfolded_Y_test)
    encoded_Y = encoder.transform(unfolded_Y_test)
    dummy_y_test = np_utils.to_categorical(encoded_Y)
    return padding_X_train, dummy_y, padding_X_test, dummy_y_test
