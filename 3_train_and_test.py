# 기계학습 모델 

import numpy as np
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from data_handler import *
from neural_design import *

import argparse
from configparser import ConfigParser
import pickle
import random
parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
config = ConfigParser()

####################################
""" PARSING STETTING """
####################################
##### argparse
####################################
# bag-of-n-grams
parser.add_argument('--use_fa', action='store_true')
parser.add_argument('--use_fb', action='store_true')
parser.add_argument('--use_fc', action='store_true')
parser.add_argument('--use_fd', action='store_true')

# model choose
parser.add_argument('--model_lr', action='store_true')
parser.add_argument('--model_pa', action='store_true')
parser.add_argument('--model_nb', action='store_true')
parser.add_argument('--model_knn', action='store_true')
parser.add_argument('--model_dt', action='store_true')
parser.add_argument('--model_svm', action='store_true')
parser.add_argument('--model_et', action='store_true')
parser.add_argument('--model_rf', action='store_true')
parser.add_argument('--model_vc', action='store_true')
parser.add_argument('--model_fnn', action='store_true')
parser.add_argument('--model_cnn', action='store_true')
parser.add_argument('--model_rnn', action='store_true')

use_fa = parser.parse_args().use_fa
use_fb = parser.parse_args().use_fb
use_fc = parser.parse_args().use_fc
use_fd = parser.parse_args().use_fd

model_lr = parser.parse_args().model_lr
model_pa = parser.parse_args().model_pa
model_nb = parser.parse_args().model_nb
model_knn = parser.parse_args().model_knn
model_dt = parser.parse_args().model_dt
model_svm = parser.parse_args().model_svm
model_et = parser.parse_args().model_et
model_rf = parser.parse_args().model_rf
model_vc = parser.parse_args().model_vc
model_fnn = parser.parse_args().model_fnn
model_cnn = parser.parse_args().model_cnn
model_rnn = parser.parse_args().model_rnn

if parser.parse_args().model_lr == True: MODEL = 'model_lr'
elif parser.parse_args().model_pa == True: MODEL = 'model_pa'
elif parser.parse_args().model_nb == True: MODEL = 'model_nb'
elif parser.parse_args().model_knn == True: MODEL = 'model_knn'
elif parser.parse_args().model_dt == True: MODEL = 'model_dt'
elif parser.parse_args().model_svm == True: MODEL = 'model_svm'
elif parser.parse_args().model_et == True: MODEL = 'model_et'
elif parser.parse_args().model_rf == True: MODEL = 'model_rf'
elif parser.parse_args().model_vc == True: MODEL = 'model_vc'
elif parser.parse_args().model_fnn == True: MODEL = 'model_fnn'
elif parser.parse_args().model_cnn == True: MODEL = 'model_cnn'
elif parser.parse_args().model_rnn == True: MODEL = 'model_rnn'
else:
    print("[arg error!] please add at least one model argument")
    exit()

##### ConfigParser
####################################
config.read('parameters.ini')
FEATURE_PATH = config.get('1_data_preparation', 'feature_path')
LR_C = config.getfloat('3_train_and_test', 'LR_C')
PA_C = config.getint('3_train_and_test', 'PA_C')
ET_n_estimators = config.getint('3_train_and_test', 'ET_n_estimators')
RF_n_estimators = config.getint('3_train_and_test', 'RF_n_estimators')
KNN_n_neighbors = config.getint('3_train_and_test', 'KNN_n_neighbors')
SVM_C = config.getint('3_train_and_test', 'SVM_C')

hidden_dim_FNN = config.getint('3_train_and_test', 'hidden_dim_FNN')
n_epoch_FNN = config.getint('3_train_and_test', 'n_epoch_FNN')
batch_size_FNN = config.getint('3_train_and_test', 'batch_size_FNN')
n_epoch_CNN = config.getint('3_train_and_test', 'n_epoch_CNN')
batch_size_CNN = config.getint('3_train_and_test', 'batch_size_CNN')
hidden_dim_RNN = config.getint('3_train_and_test', 'hidden_dim_RNN')
n_epoch_RNN = config.getint('3_train_and_test', 'n_epoch_RNN')
batch_size_RNN = config.getint('3_train_and_test', 'batch_size_RNN')

######################################################
print('\n[Load feature-vec and choose feature set]')
######################################################

### load data
ann_info = data_load(None)
temp_list = load_feature_vec(FEATURE_PATH)
Ytrain = temp_list[8]
Ytest = temp_list[9]

### choose feature set
Xtrain, Xtest = choice_feature_set_v2(use_fa, use_fb, use_fc, use_fd, temp_list)
print('>> selected feature vector dim = ', len(Xtrain[0][0]))

################################################################
print('\n[Suitable Vector-form Transformation to each model]')
################################################################
if MODEL == 'model_fnn':
    X_train_F, Y_train_F, X_test, Y_test_K = prepare_for_FNN(Xtrain, Ytrain, Xtest, Ytest)
elif MODEL == 'model_cnn':
    X_train_C, Y_train_C, X_test, Y_test_K = prepare_for_CNN(Xtrain, Ytrain, Xtest, Ytest)
elif MODEL == 'model_rnn':
    X_train_R, Y_train_R, X_test, Y_test_K = prepare_for_RNN(Xtrain, Ytrain, Xtest, Ytest)
else: # scikit learn models
    X_train, Y_train = unfold_data(Xtrain, Ytrain)
    X_test, Y_test = unfold_data(Xtest, Ytest)
    #assert(len(X_train) == len(Y_train))
    #assert(len(X_test[0]) == len(Y_test[0]))
    #assert(len(X_train) == len(Y_train))
    #assert(len(X_test[0]) == len(Y_test[0]))

##############################
print('\n[Train Model]')
##############################
### set model
if MODEL=='model_lr': sel_model = LogisticRegression(penalty='l2', C=LR_C)
elif MODEL=='model_pa': sel_model = PassiveAggressiveClassifier(C=PA_C)
elif MODEL=='model_nb': sel_model = GaussianNB()
elif MODEL=='model_knn': sel_model = KNeighborsClassifier(n_neighbors=KNN_n_neighbors)
elif MODEL=='model_dt': sel_model = DecisionTreeClassifier()
elif MODEL=='model_svm': sel_model = SVC(C=SVM_C)
elif MODEL=='model_et': sel_model = ExtraTreesClassifier(n_estimators=ET_n_estimators)
elif MODEL=='model_rf': sel_model = RandomForestClassifier(n_estimators=RF_n_estimators)
elif MODEL=='model_vc':
    estimators = []
    estimators.append(('lr', LogisticRegression(penalty='l2', C=LR_C) ))
    estimators.append(('et', ExtraTreesClassifier(n_estimators=ET_n_estimators) ))
    estimators.append(('svm', SVC(C=SVM_C)))
    sel_model = VotingClassifier(estimators)
elif MODEL=='model_fnn':
    input_dim = len(X_train_F[0])
    output_dim = len(Y_train_F[0])
    sel_model = feedforward(input_dim, hidden_dim_FNN, output_dim)
elif MODEL=='model_cnn':
    x_square = X_train_C.shape[3]
    output_dim = Y_train_C.shape[1]
    sel_model = convolutional(x_square, output_dim)
elif MODEL=='model_rnn':
    time_length = X_train_R.shape[1]
    feature_dim = X_train_R.shape[2]
    output_dim = Y_train_R.shape[1]
    sel_model = recurrent(feature_dim, hidden_dim_RNN, output_dim, time_length)

### train model
if MODEL=='model_fnn':
    sel_model.fit(X_train_F, Y_train_F, epochs=n_epoch_FNN, batch_size=batch_size_FNN, verbose=0)
elif MODEL=='model_cnn':
    sel_model.fit(X_train_C, Y_train_C, epochs=n_epoch_CNN, batch_size=batch_size_CNN, verbose=0)
elif MODEL=='model_rnn':
    sel_model.fit(X_train_R, Y_train_R ,epochs=n_epoch_RNN, batch_size=batch_size_RNN, verbose=0)
else:
    sel_model.fit(X_train, Y_train)

##############################
print('\n[Evaluate Model]')
##############################
if MODEL=='model_fnn' or MODEL=='model_cnn' or MODEL=='model_rnn':
    predictions = sel_model.predict_classes(X_test, verbose=0)
    Y_test = dummy_to_integer(Y_test_K)
else:
    predictions = sel_model.predict(X_test)

print('Evaluation Result: ', MODEL)
print('* acc: ', accuracy_score(Y_test, predictions))
#print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions, target_names=list(ann_info.values())))

# micro/macro averaged scores
pre, rec, f1, sup = (precision_recall_fscore_support(Y_test, predictions, average='micro'))
print('* micro averaged scores: ', round(pre, 3), '(precision)', '\t', round(rec, 3), '(recall)', '\t', round(f1, 3), '(f1)')
pre, rec, f1, sup = (precision_recall_fscore_support(Y_test, predictions, average='macro'))
print('* macro averaged scores: ', round(pre, 3), '(precision)', '\t', round(rec, 3), '(recall)', '\t', round(f1, 3), '(f1)')
