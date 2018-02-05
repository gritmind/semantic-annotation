


from feature_design import *

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
parser.add_argument('--use_1gram', action='store_true')
parser.add_argument('--use_2gram', action='store_true')
parser.add_argument('--use_3gram', action='store_true')
parser.add_argument('--use_5gram', action='store_true')
parser.add_argument('--use_classes_1gram', action='store_true')
# stanford parser
parser.add_argument('--use_stanford_pos', action='store_true')
parser.add_argument('--use_stanford_parser', action='store_true')
# Rule
parser.add_argument('--use_rule', action='store_true')
# spaCy parser
parser.add_argument('--use_spacy_pos', action='store_true')
parser.add_argument('--use_spacy_chunk', action='store_true')
parser.add_argument('--use_spacy_parser', action='store_true')

use_1gram = parser.parse_args().use_1gram
use_2gram = parser.parse_args().use_2gram
use_3gram = parser.parse_args().use_3gram
use_5gram = parser.parse_args().use_5gram
use_classes_1gram = parser.parse_args().use_classes_1gram
use_stanford_pos = parser.parse_args().use_stanford_pos
use_stanford_parser = parser.parse_args().use_stanford_parser
use_rule = parser.parse_args().use_rule
use_spacy_pos = parser.parse_args().use_spacy_pos
use_spacy_chunk = parser.parse_args().use_spacy_chunk
use_spacy_parser = parser.parse_args().use_spacy_parser

##### ConfigParser
####################################
config.read('parameters.ini')
LOOKUP_TABLE_PATH = config.get('1_data_preparation', 'lookup_table_path')
TEMP_DATA_PATH = config.get('1_data_preparation', 'temp_data_path')
FEATURE_PATH = config.get('1_data_preparation', 'feature_path')

# bog-of-n-grams
window_size_1gram = config.getint('2_feature_extraction', 'window_size_1gram')
window_size_class = config.getint('2_feature_extraction', 'window_size_class')
window_size_1gram = (window_size_1gram * 2) + 1
window_size_class = (window_size_class * 2) + 1

# stanford
window_size_pos = config.getint('2_feature_extraction', 'window_size_pos')
window_size_depth = config.getint('2_feature_extraction', 'window_size_depth')
window_size_n_siblings = config.getint('2_feature_extraction', 'window_size_n_siblings')
window_size_pos = (window_size_pos * 2) + 1
window_size_depth = (window_size_depth * 2) + 1
window_size_n_siblings = (window_size_n_siblings * 2) + 1

# spaCy
window_size_spacy_pos = config.getint('2_feature_extraction', 'window_size_spacy_pos')
window_size_spacy_pos = (window_size_spacy_pos * 2) + 1

##############################################
print('\n[Load temp-data & lookup-tables]')
##############################################

### load data
temp_data_list = load_temp_data(TEMP_DATA_PATH)
pre_X_train = temp_data_list[0]
pre_X_test = temp_data_list[1]
Xtrain = temp_data_list[2]
Xtest = temp_data_list[3]
Ytrain = temp_data_list[4]
Ytest = temp_data_list[5]

### load lookup-tables
lookup_table_list = load_lookup_tables(LOOKUP_TABLE_PATH)
lookup_1_gram = lookup_table_list[0]
lookup_2_gram = lookup_table_list[1]
lookup_3_gram = lookup_table_list[2]
lookup_5_gram = lookup_table_list[3]
lookup_classes_1gram = lookup_table_list[4]


###############################################
print('\n[Bag-of-n-grams feature extraction]')
###############################################

### initalize feature vector
FE_X_train, FE_X_test = init_feature_vec(pre_X_train, pre_X_test)

### add bag-of-n-grams features
if use_1gram==True:
    FE_X_train, FE_X_test = add_1gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_1_gram, window_size_1gram)
if use_2gram==True:
    FE_X_train, FE_X_test = add_2gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_2_gram)
if use_3gram==True:
    FE_X_train, FE_X_test = add_3gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_3_gram)
if use_5gram==True:
    FE_X_train, FE_X_test = add_5gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_5_gram)
if use_classes_1gram==True:
    FE_X_train, FE_X_test = add_classes_1gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_classes_1gram, window_size_class)

### save
dump(FE_X_train, FEATURE_PATH+'FE_X_train')
dump(FE_X_test, FEATURE_PATH+'FE_X_test')
print('>> Total bag-of-n-grams feature dim =', len(FE_X_train[0][0]))

################################################################
print('\n[Standford (constituency parser) feature extraction]')
################################################################

### initalize feature vector
FE2_X_train, FE2_X_test = init_feature_vec(Xtrain, Xtest)

### add Standford (constituency parser) features
if use_stanford_pos==True:
    lookup_pos = create_lookup_pos()
    FE2_X_train, FE2_X_test = add_pos(FE2_X_train, FE2_X_test, Xtrain, Xtest, lookup_pos, window_size_pos)
if use_stanford_parser==True:
    FE2_X_train, FE2_X_test = add_constituency(FE2_X_train, FE2_X_test, Xtrain, Xtest, window_size_depth, window_size_n_siblings)

### save
dump(FE2_X_train, FEATURE_PATH+'FE2_X_train')
dump(FE2_X_test, FEATURE_PATH+'FE2_X_test')
print('>> Total Standford (constituency parser) feature dim =', len(FE2_X_train[0][0]))

############################################
print('\n[Rule feature extraction]')
############################################

### initalize feature vector
FE3_X_train, FE3_X_test = init_feature_vec(pre_X_train, pre_X_test)

### add rule features
if use_rule==True:
    FE3_X_train, FE3_X_test = add_rule(FE3_X_train, FE3_X_test, pre_X_train, pre_X_test)

### save
dump(FE3_X_train, FEATURE_PATH+'FE3_X_train')
dump(FE3_X_test, FEATURE_PATH+'FE3_X_test')
print('>> Total Rule feature dim =', len(FE3_X_train[0][0]))

#########################################################
print('\n[spaCy (dependency parser) feature extraction]')
#########################################################

### initalize feature vector
FE4_X_train, FE4_X_test = init_feature_vec(Xtrain, Xtest)

### add spaCy (dependency parser) features
if use_spacy_pos==True:
    lookup_spacy_pos = create_lookup_spacy_pos()
    FE4_X_train, FE4_X_test = add_spacy_pos(FE4_X_train, FE4_X_test, Xtrain, Xtest, lookup_spacy_pos, window_size_spacy_pos)
if use_spacy_chunk==True:
    FE4_X_train, FE4_X_test = add_spacy_chunk(FE4_X_train, FE4_X_test, Xtrain, Xtest)
if use_spacy_parser==True:
    lookup_spacy_pos = create_lookup_spacy_pos()
    lookup_spacy_dependency = create_lookup_spacy_dependency()
    FE4_X_train, FE4_X_test = add_spacy_parser(FE4_X_train, FE4_X_test, Xtrain, Xtest, lookup_spacy_pos, lookup_spacy_dependency)

### save
dump(FE4_X_train, FEATURE_PATH+'FE4_X_train')
dump(FE4_X_test, FEATURE_PATH+'FE4_X_test')
print('>> Total spaCy (dependency parser) feature dim =', len(FE4_X_train[0][0]))

### closing-remarks
dump(Ytrain, FEATURE_PATH+'Ytrain')
dump(Ytest, FEATURE_PATH+'Ytest')
print('\nTotal feature vector dim = ', len(FE_X_train[0][0])+len(FE2_X_train[0][0])+len(FE3_X_train[0][0])+len(FE4_X_train[0][0]))
