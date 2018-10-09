# 데이터 전처리, 어휘 사전, lookup table 만들기 등 재료를 준비하는 단계

import argparse
from configparser import ConfigParser
import pickle
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
config = ConfigParser()

import numpy as np
import pandas as pd
import operator
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import words
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from collections import Counter
import matplotlib.pyplot as plt
from data_handler import *
from nltk.corpus import wordnet as wn

lemmatiser = WordNetLemmatizer() # WordNet Lemmatizer

####################################
""" PARSING STETTING """
####################################
##### argparse
####################################
#parser.add_argument("-f", "--foo", required=True)
#args = parser.parse_args()
#foo = args.foo

# STEM_TOOL
parser.add_argument('--stemming', action='store_true')
parser.add_argument('--lemmatization', action='store_true')

# dataset argument parsing
if parser.parse_args().stemming == True:
    STEM_TOOL = 'stemming'
elif parser.parse_args().yelpp == True:
    STEM_TOOL = 'lemmatization'
else:
    print("[arg error!] please add at least one dataset argument")
    exit()

##### ConfigParser
####################################
config.read('parameters.ini')
FREQ_DIC_PATH = config.get('1_data_preparation', 'freq_dic_path')
LOOKUP_TABLE_PATH = config.get('1_data_preparation', 'lookup_table_path')
TEMP_DATA_PATH = config.get('1_data_preparation', 'temp_data_path')
#bool_val = config.getboolean('section_a', 'bool_val')
seed = config.getint('1_data_preparation', 'seed')
VAL_RATIO = config.getfloat('1_data_preparation', 'val_ratio')

thr_1gram = config.getint('1_data_preparation', 'thr_1gram')
thr_2gram = config.getint('1_data_preparation', 'thr_2gram')
thr_3gram = config.getint('1_data_preparation', 'thr_3gram')
thr_5gram = config.getint('1_data_preparation', 'thr_5gram')

thr_component = config.getint('1_data_preparation', 'thr_component')
thr_refinement_of_component = config.getint('1_data_preparation', 'thr_refinement_of_component')
thr_action = config.getint('1_data_preparation', 'thr_action')
thr_refinement_of_action = config.getint('1_data_preparation', 'thr_refinement_of_action')
thr_condition = config.getint('1_data_preparation', 'thr_condition')
thr_priority = config.getint('1_data_preparation', 'thr_priority')
thr_motivation = config.getint('1_data_preparation', 'thr_motivation')
thr_role = config.getint('1_data_preparation', 'thr_role')
thr_object = config.getint('1_data_preparation', 'thr_object')
thr_refinement_of_object = config.getint('1_data_preparation', 'thr_refinement_of_object')
thr_sub_action = config.getint('1_data_preparation', 'thr_sub_action')
thr_sub_argument_of_action = config.getint('1_data_preparation', 'thr_sub_argument_of_action')
thr_sub_priority = config.getint('1_data_preparation', 'thr_sub_priority')
thr_sub_role = config.getint('1_data_preparation', 'thr_sub_role')
thr_sub_object = config.getint('1_data_preparation', 'thr_sub_object')
thr_sub_refinement_of_object = config.getint('1_data_preparation', 'thr_sub_refinement_of_object')
thr_none = config.getint('1_data_preparation', 'thr_none')


########################################
print('\n[Load Data and Data Cleaning]')
########################################

### DATA LOAD
file_path = 'dataset/labeled-requirements.txt'
X, Y, ann_info = data_load(file_path)

### DATA STATISTIC ANALYSIS
#show_data_statistic(Y, ann_info) # print # of sentences per class

### DATA CLEANING
X, Y = data_cleaning(X, Y)

### SPLIT DATASET
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=VAL_RATIO, random_state=seed)
print('>> (Split) Train:', len(Xtrain), ', Test:',len(Xtest))

####################################
print('\n[Preprocessing]')
####################################

pre_X_train = Xtrain[:]
pre_X_test = Xtest[:]
#Y_train = Ytrain[:]

### lowering
pre_X_train, pre_X_test = lowering(pre_X_train, pre_X_test)

### lemmatization or stemming
pre_X_train, pre_X_test = stem_lemma(pre_X_train, pre_X_test, STEM_TOOL)

###############################################################
print('\n[Build Vocab and Save Freq Dict]') # only using train set
###############################################################

### corpus based
vocab_1_gram = build_vocab_1_gram(pre_X_train, thr_1gram, FREQ_DIC_PATH)
vocab_2_gram = build_vocab_2_gram(pre_X_train, thr_2gram, FREQ_DIC_PATH)
vocab_3_gram = build_vocab_3_gram(pre_X_train, thr_3gram, FREQ_DIC_PATH)
vocab_5_gram = build_vocab_5_gram(pre_X_train, thr_5gram, FREQ_DIC_PATH)

### limit to class not corpus
vocab_classes_list = build_vocab_1gram_classes(pre_X_train, Ytrain, thr_1gram, FREQ_DIC_PATH, ann_info)

############################################
print('[Convert Vocab into Lookup Table]')
############################################
# add unknown token

lookup_1_gram = convert_1_gram_lookup(vocab_1_gram)
lookup_2_gram = convert_2_gram_lookup(vocab_2_gram)
lookup_3_gram = convert_3_gram_lookup(vocab_3_gram)
lookup_5_gram = convert_5_gram_lookup(vocab_5_gram)

lookup_classes_1gram = convert_classes_1gram_lookup(vocab_classes_list)

# X_train = pre_X_train
# X_test = pre_X_test

########################################################
print('\n[Save all lookup tables and preprocessed data]')
########################################################

LOOKUP_TABLE_PATH
TEMP_DATA_PATH

dump(lookup_1_gram, LOOKUP_TABLE_PATH+'lookup_1_gram')
dump(lookup_2_gram, LOOKUP_TABLE_PATH+'lookup_2_gram')
dump(lookup_3_gram, LOOKUP_TABLE_PATH+'lookup_3_gram')
dump(lookup_5_gram, LOOKUP_TABLE_PATH+'lookup_5_gram')
dump(lookup_classes_1gram, LOOKUP_TABLE_PATH+'lookup_classes_1gram')

dump(Xtrain, TEMP_DATA_PATH+'Xtrain')
dump(Ytrain, TEMP_DATA_PATH+'Ytrain')
dump(Xtest, TEMP_DATA_PATH+'Xtest')
dump(Ytest, TEMP_DATA_PATH+'Ytest')

dump(pre_X_train, TEMP_DATA_PATH+'pre_X_train')
dump(pre_X_test, TEMP_DATA_PATH+'pre_X_test')
