# 데이터 로드, 출력, 전처리 등과 같은 데이터 처리를 위한 함수 집합소

import pickle
import random
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
seed = 7
stemmer = PorterStemmer()

####################################################################
####################################################################
""" for 3_train_and_test.py """
####################################################################
####################################################################

def load_feature_vec(FEATURE_PATH):
    FE_X_train = load(FEATURE_PATH+'FE_X_train')
    FE_X_test = load(FEATURE_PATH+'FE_X_test')
    FE2_X_train = load(FEATURE_PATH+'FE2_X_train')
    FE2_X_test = load(FEATURE_PATH+'FE2_X_test')
    FE3_X_train = load(FEATURE_PATH+'FE3_X_train')
    FE3_X_test = load(FEATURE_PATH+'FE3_X_test')
    FE4_X_train = load(FEATURE_PATH+'FE4_X_train')
    FE4_X_test = load(FEATURE_PATH+'FE4_X_test')
    Ytrain = load(FEATURE_PATH+'Ytrain')
    Ytest = load(FEATURE_PATH+'Ytest')
    return [FE_X_train, FE_X_test, FE2_X_train, FE2_X_test, FE3_X_train, FE3_X_test, FE4_X_train, FE4_X_test, Ytrain, Ytest]

####################################################################
####################################################################
""" for 1_data_preparation.py """
####################################################################
####################################################################

##########################################
""" Convert Vocab into Lookup Table """
##########################################

def convert_1_gram_lookup(vocab_1_gram):
    word_one_gram_lookup_table = { }
    for i, token in enumerate(vocab_1_gram):
        word_one_gram_lookup_table[token] = i+1 # 1부터 시작하게 유도
    # Add unknown token
    word_one_gram_lookup_table['unknown_token'] = len(word_one_gram_lookup_table)+1
    return word_one_gram_lookup_table

def convert_2_gram_lookup(vocab_2_gram):
    word_bi_gram_lookup_table = { }
    for i, token in enumerate(vocab_2_gram):
        word_bi_gram_lookup_table[token] = i+1
    # Add unknown token
    word_bi_gram_lookup_table[('unknown_token', 'unknown_token')] = len(word_bi_gram_lookup_table)+1
    return word_bi_gram_lookup_table

def convert_3_gram_lookup(vocab_3_gram):
    word_tri_gram_lookup_table = { }
    for i, token in enumerate(vocab_3_gram):
        word_tri_gram_lookup_table[token] = i+1
    # Add unknown token
    word_tri_gram_lookup_table['unknown_token', 'unknown_token', 'unknown_token'] = len(word_tri_gram_lookup_table)+1
    return word_tri_gram_lookup_table

def convert_5_gram_lookup(vocab_5_gram):
    ### Create word_five_gram Lookup Table
    word_five_gram_lookup_table = { }
    for i, token in enumerate(vocab_5_gram):
        word_five_gram_lookup_table[token] = i+1
    # Add unknown token
    word_five_gram_lookup_table['unknown_token', 'unknown_token', 'unknown_token', 'unknown_token', 'unknown_token'] = len(word_five_gram_lookup_table)+1
    return word_five_gram_lookup_table

def convert_classes_1gram_lookup(vocab_classes_list):
    lookup_classes_1gram = []
    for every_vocab in vocab_classes_list:
        temp_1gram_lookup_table = { }
        for i, token in enumerate(every_vocab):
            temp_1gram_lookup_table[token] = i+1
        # Add unknown token
        temp_1gram_lookup_table['unknown_token'] = len(temp_1gram_lookup_table)+1
        lookup_classes_1gram.append(temp_1gram_lookup_table)
    return lookup_classes_1gram


#############################################################
""" Build Vocab and Save Freq Dict """ # only using train set
#############################################################

# 1-gram Vocabulary
def build_vocab_1_gram(X_train, thr_1gram, FREQ_DIC_PATH):

    """ Define unknown token using only training data """
    ## 데이터 셋이 작아서 그런지 생각보다 freq 1인 단어들이 많다.
    ## stemming말고 lemmatisation을 통해 word 어원의 형태를 잡고, wordnet과 같은 외부 자원을 사용해서
    ## freq 1인 단어라도 wordnet에 해당되는 단어는 unknown token으로 처리하지 않는 방식도 생각해보자.
    ## Stemming 후, word freq 1 비율: ('ratio of unknown_word:', 48.505434782608695)

    # Word frequency dictionary
    word_count_dic = dict()
    for i, sentence in enumerate(X_train):
        for j, token in enumerate(sentence):
            word_count_dic[token] = word_count_dic.get(token, 0) + 1

    #sorted(word_count_dic.items(), key=lambda x:x[1], reverse=True)
    #sorted(word_count_dic.items(), key=lambda x:x[1])

    # Write Frequency Distribution
    text_file = open(FREQ_DIC_PATH+"vocab_1gram.txt", "w")
    for w in sorted(word_count_dic, key=word_count_dic.get, reverse=True):
        text_file.write(w)
        text_file.write('\t')
        text_file.write(str(word_count_dic[w]))
        text_file.write('\n')
    text_file.close()

    # Build unknwon word list based on frequency or wordnet (e.g., threshold: freq 1)
    unknown_word_list = []
    word_one_gram_voca_set = []
    for key, value in word_count_dic.items():
        if value <= thr_1gram: # if only 1 frequency
            unknown_word_list.append(key)
        else:
            word_one_gram_voca_set.append(key)
    print(">> Ratio of 1-gram unknown word:", round( len(unknown_word_list) / float(len(word_count_dic)) * 100, 2))

    # """ Replace unknown token using only training data """
    # for i, sentence in enumerate(X_train):
    #     for j, token in enumerate(sentence):
    #         if any(token in t for t in unknown_word_list):
    #             X_train[i][j] = 'unknown_token'
    print('>> 1-gram vocab size: ', len(word_one_gram_voca_set))
    return word_one_gram_voca_set

def build_vocab_2_gram(X_train, thr_2gram, FREQ_DIC_PATH):

    ### create bi-gram set
    bigram_list = []
    for i, sentence in enumerate(X_train):
        # Extract bigrams from a single sentence
        bgs = nltk.ngrams(sentence, 2)
        # Compute frequency distribution
        output = nltk.FreqDist(bgs).items()
        # Integrate All
        bigram_list += output

    ### merge the same tuple
    merged_bigram_freq_list = []
    cnt = 0
    for i, bigram_and_freq in enumerate(bigram_list):
        check = False
        for j, merged_bigram_and_freq in enumerate(merged_bigram_freq_list):
            if bigram_and_freq[0] == merged_bigram_and_freq[0]:
    #             print 'here', bigram_and_freq[0], merged_bigram_and_freq[0]
                merged_bigram_freq_list[j][1] = merged_bigram_and_freq[1] + bigram_and_freq[1] # merging freqeucny
                check = True
        if check == False:
            temp_list = []
            temp_list.append(bigram_and_freq[0])
            temp_list.append(bigram_and_freq[1])
            merged_bigram_freq_list.append(temp_list)
    #     cnt += 1
    #     if cnt==50:
    #         break
    # print 'total # of bi-gram:', len(merged_bigram_freq_list)
    # print merged_bigram_freq_list
    # print np.array(merged_bigram_freq_list).shape

    ### Write Frequency Distribution
    text_file = open(FREQ_DIC_PATH+"vocab_2gram.txt", "w")
    for bigram, freq in sorted(merged_bigram_freq_list, key=lambda x: x[1], reverse=True):
        text_file.write(str(bigram))
        text_file.write('\t')
        text_file.write(str(freq))
        text_file.write('\n')
    text_file.close()

    ### Filter and make bigram voca set
    word_bi_gram_voca_set = []
    for bigram, freq in merged_bigram_freq_list:
        if freq > thr_2gram:
            word_bi_gram_voca_set.append(bigram)
    print('>> 2-gram vocab size: ', len(word_bi_gram_voca_set))
    return word_bi_gram_voca_set

def build_vocab_3_gram(X_train, thr_3gram, FREQ_DIC_PATH):
    ### create tri-gram set
    trigram_list = []
    for i, sentence in enumerate(X_train):
        # Extract trigrams from a single sentence
        tgs = nltk.ngrams(sentence, 3)
        # Compute frequency distribution
        output = nltk.FreqDist(tgs).items()
        # Integrate All
        trigram_list += output

    ### merge the same tuple
    merged_trigram_freq_list = []
    cnt = 0
    for i, trigram_and_freq in enumerate(trigram_list):
        check = False
        for j, merged_trigram_and_freq in enumerate(merged_trigram_freq_list):
            if trigram_and_freq[0] == merged_trigram_and_freq[0]:
    #             print 'here', bigram_and_freq[0], merged_bigram_and_freq[0]
                merged_trigram_freq_list[j][1] = merged_trigram_and_freq[1] + trigram_and_freq[1] # merging freqeucny
                check = True

        if check == False:
            temp_list = []
            temp_list.append(trigram_and_freq[0])
            temp_list.append(trigram_and_freq[1])
            merged_trigram_freq_list.append(temp_list)

    #     cnt += 1
    #     if cnt==50:
    #         break

    # print 'total # of bi-gram:', len(merged_bigram_freq_list)
    # print merged_bigram_freq_list
    # print np.array(merged_bigram_freq_list).shape

    ### Write Frequency Distribution
    text_file = open(FREQ_DIC_PATH+"vocab_3gram.txt", "w")
    for trigram, freq in sorted(merged_trigram_freq_list, key=lambda x: x[1], reverse=True):
        text_file.write(str(trigram))
        text_file.write('\t')
        text_file.write(str(freq))
        text_file.write('\n')
    text_file.close()

    ### Filter and make trigram voca set
    word_tri_gram_voca_set = []

    for trigram, freq in merged_trigram_freq_list:
        if freq > thr_3gram:
            word_tri_gram_voca_set.append(trigram)
    print('>> 3-gram vocab size: ', len(word_tri_gram_voca_set))
    return word_tri_gram_voca_set

def build_vocab_5_gram(X_train, thr_5gram, FREQ_DIC_PATH):
    ### create five-gram set
    fivegram_list = []
    for i, sentence in enumerate(X_train):

        # Extract trigrams from a single sentence
        fgs = nltk.ngrams(sentence, 5)

        # Compute frequency distribution
        output = nltk.FreqDist(fgs).items()

        # Integrate All
        fivegram_list += output

    ### merge the same tuple
    merged_fivegram_freq_list = []
    cnt = 0
    for i, fivegram_and_freq in enumerate(fivegram_list):

        check = False
        for j, merged_fivegram_and_freq in enumerate(merged_fivegram_freq_list):

            if fivegram_and_freq[0] == merged_fivegram_and_freq[0]:
    #             print 'here', bigram_and_freq[0], merged_bigram_and_freq[0]
                merged_fivegram_freq_list[j][1] = merged_fivegram_and_freq[1] + fivegram_and_freq[1] # merging freqeucny
                check = True

        if check == False:
            temp_list = []
            temp_list.append(fivegram_and_freq[0])
            temp_list.append(fivegram_and_freq[1])
            merged_fivegram_freq_list.append(temp_list)

    #     cnt += 1
    #     if cnt==50:
    #         break

    # print 'total # of bi-gram:', len(merged_bigram_freq_list)
    # print merged_bigram_freq_list
    # print np.array(merged_bigram_freq_list).shape

    ### Write Frequency Distribution
    text_file = open(FREQ_DIC_PATH+"vocab_5gram.txt", "w")
    for fivegram, freq in sorted(merged_fivegram_freq_list, key=lambda x: x[1], reverse=True):
        text_file.write(str(fivegram))
        text_file.write('\t')
        text_file.write(str(freq))
        text_file.write('\n')
    text_file.close()

    ### Filtering out
    word_five_gram_voca_set = []
    for fivegram, freq in merged_fivegram_freq_list:
        if freq > thr_5gram:
            word_five_gram_voca_set.append(fivegram)
    print('>> 5-gram voca size: ', len(word_five_gram_voca_set))
    return word_five_gram_voca_set

### Write out
def dict_wirte_to_txt(class_list, file_name):
    text_file = open(file_name, "w")
    class_dict = dict(Counter(class_list))
    for w in sorted(class_dict, key=class_dict.get, reverse=True):
        text_file.write(w)
        text_file.write('\t')
        text_file.write(str(class_dict[w]))
        text_file.write('\n')
    text_file.close()
    return class_dict

### Filtering out and make voca set
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def filtering_out_for_dict(dict, thr_1gram):
    temp_voca_set = []
    for key, value in dict.items():
        if value > thr_1gram: # if only 1 frequency
            temp_voca_set.append(key)
    return temp_voca_set

def build_vocab_1gram_classes(X_train, Y_train, thr_1gram, FREQ_DIC_PATH, ann_info):
    # word count list
    word_count_classes = [[] for i in range(len(ann_info))]
    for i, sentence in enumerate(Y_train): # y label
        for j, token in enumerate(sentence):
            for k in range(0, len(ann_info)):
                if token == k:
                    word_count_classes[k].append(X_train[i][j]) # x data
    # dict list
    dict_classes = [None] * len(ann_info)
    for k in range(0, len(ann_info)):
        dict_classes[k] = dict_wirte_to_txt(word_count_classes[k], FREQ_DIC_PATH+str(ann_info[k])+'.txt')
    # vocab list
    vocab_classes = [None] * len(ann_info)
    for k in range(0, len(ann_info)):
        vocab_classes[k] = filtering_out_for_dict(dict_classes[k], thr_1gram)
    print('>> 1gram vocabs of classes are saved.')
    return vocab_classes




####################################
""" Preprocessing """
####################################
### LOWERING
def lowering(X_train, X_test):
    for i, sentence in enumerate(X_train): # for training data
        for j, token in enumerate(sentence):
            X_train[i][j] = token.lower()
    for i, sentence in enumerate(X_test): # for testing data
        for j, token in enumerate(sentence):
            X_test[i][j] = token.lower()
    return X_train, X_test

### Lemmatization Description
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']
def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']
def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS', 'IN'] # IN 추가 IN중에서 as가 그냥 lemmitization되면 a가 되므로, 'a'로 설정해준다.
def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None
def lemmatise(tuple):
# to distinguish whether token is noun or verb
# because in lemmatization, there are different result according to them
    verb_tag_set = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # verb tag list from 'nltk.help.upenn_tagset()'
    token = tuple[0]
    pos_tag = tuple[1]
    if penn_to_wn(pos_tag) == None:
        return str(lemmatiser.lemmatize(token))
    else:
        return str(lemmatiser.lemmatize(token, penn_to_wn(pos_tag)))

def stem_lemma(X_train, X_test, STEM_TOOL):
    if STEM_TOOL == 'stemming':
        # 1. Stemming
        for i, sentence in enumerate(X_train): # for training data
            for j, token in enumerate(sentence):
                X_train[i][j] = str(stemmer.stem(token))
        for i, sentence in enumerate(X_test): # for testing data
            for j, token in enumerate(sentence):
                X_test[i][j] = str(stemmer.stem(token))
    elif STEM_TOOL == 'lemmatization':
        # 2. Lemmatise
        for i, sentence in enumerate(X_train): # for training data
            pos_sentence = nltk.pos_tag(sentence)
            for j, token in enumerate(sentence):
                X_train[i][j] = lemmatise(pos_sentence[j]) # input: tuple
        for i, sentence in enumerate(X_test): # for training data
            pos_sentence = nltk.pos_tag(sentence)
            for j, token in enumerate(sentence):
                X_test[i][j] = lemmatise(pos_sentence[j]) # input: tuple
    return X_train, X_test



####################################
""" Load Data and Data Cleaning """
####################################

### DATA STATISTIC ANALYSIS
def tkn_assign(y, boolean_list):
    boolean_list[y] = True
    pass
def increment_cnt(sent_num_list, boolean_list):
    for i in range(0, len(sent_num_list)):
        if boolean_list[i] == True:
            sent_num_list[i] += 1
            boolean_list[i] = False
    pass
def show_data_statistic(Y, ann_info):
    ### print # of sentences per class
    ann_info = {val:key for (key, val) in ann_info.items()} # key-value switching
    boolean_list = [False] * 17
    sent_num_list = [0] * 17
    for i, sent in enumerate(Y):
        for j, token in enumerate(sent):
            tkn_assign(Y[i][j], boolean_list)
        increment_cnt(sent_num_list, boolean_list)
    print('>> # of sentences per class')
    for i in range(0, len(sent_num_list)):
        print(sent_num_list[i], '\t', ann_info[i])
    pass

### DATA CLEANING
def data_cleaning(X, Y):
    # change word
    for i, sentence in enumerate(X):
        for j, token in enumerate(sentence):
            if token == 'maybe':
                X[i][j] = 'may'
    cnt = 0
    for i, sentence in enumerate(X):
        for j, token in enumerate(sentence):
            if sentence[j] == '.':
                if sentence[j-1] == 'etc':
                    sentence.remove(sentence[j])  # X delete
                    del Y[i][j]
                    cnt += 1
    #print('DELETE counts = ', cnt)
    puntation_list = ['(', ')', ';']
    # 이상하게도 한 번만 for loop 돌면서 삭제하면, 완전히 다 삭제가 안된다.
    # 또 다른 이유가 있겠지만, 그냥 여기서 for loop를 2번 돌려서 삭제한다.
    # First delete
    cnt = 0
    for i, sentence in enumerate(X):
        for j, token in enumerate(sentence):
            if any(token in t for t in puntation_list):
                sentence.remove(sentence[j])  # X delete
                del Y[i][j]
                cnt += 1
    #print('(first) DELETE puntuation counts = ', cnt)
    # Second delete
    cnt = 0
    for i, sentence in enumerate(X):
        for j, token in enumerate(sentence):
            if any(token in t for t in puntation_list):
                sentence.remove(sentence[j])  # X delete
                del Y[i][j]
                cnt += 1
    #print('(second) DELETE puntuation counts = ', cnt)
    # terminator comma add!
    cnt = 0
    for i, sentence in enumerate(X):
        if not sentence[-1] == '.':
            if not sentence[-1] == '?':
                X[i] = X[i] + ['.'] # 조심: X[i] += ['.'] 사용x
                Y[i] = Y[i] + [16] # none label
                cnt += 1
    #print('ADD counts = ', cnt)
    return X, Y

### DATA LOAD
def data_load(load_file_name):
    ann_info = {0: 'component', 1: 'refinement_of_component', 2: 'action',
                3: 'refinement_of_action',
                4: 'condition', 5: 'priority', 6: 'motivation', 7: 'role',
                8: 'object', 9: 'refinement_of_object',
                10: 'sub_action', 11: 'sub_argument_of_action', 12: 'sub_priority',
                13: 'sub_role', 14: 'sub_object',
                15: 'sub_refinement_of_object', 16: 'none'}
    if load_file_name == None:
        return ann_info
    # convert key, val in dicts
    ann_info = dict((v,k) for k,v in ann_info.items())
    X = []
    Y = []
    currentX = []
    currentY = []
    split_sequences = True
    for line in open(load_file_name):
        line = line.rstrip()
        if line:
            row = line.split()
            word, tag = row
            currentX.append(word)

            tag_list = list(tag)
            del tag_list[0]
            del tag_list[-1]
            new_tag = ''.join(tag_list)

            currentY.append(ann_info[new_tag])
        elif split_sequences: # the end of sentence
            X.append(currentX)
            Y.append(currentY)
            currentX = []
            currentY = []
    print(">> The total number of sentences:", len(X))
    ### shuffle...
    merged_data = list(zip(X, Y))
    random.seed(seed)
    random.shuffle(merged_data)
    X, Y = zip(*merged_data)
    X = list(X)
    Y = list(Y)
    assert(len(X) == len(Y))
    # convert to original
    ann_info = dict((v,k) for k,v in ann_info.items())
    return X, Y, ann_info


def del_ele_in_li(delete_list, X, Y, for_X=True, one_dim=False):
### What is this function?
# In a list (1D or 2D) when we want to delete certain elements defined by 'delete_list'
### Warning
# the elements in the 'delete_list" must be string.
    if one_dim==True:
        # 1D Array
        i = 0
        while i < len(Y):
            if for_X==True:
                if any(str(X[i]) in t for t in delete_list):
                    del Y[i]
                    del X[i]
                else:
                    i+=1
            else:
                if any(str(Y[i]) in t for t in delete_list):
                    del Y[i]
                    del X[i]
                else:
                    i+=1
    else:
        # 2D Array
        for i, sentence in enumerate(Y): # row scanning
            j = 0
            while j < len(sentence):
                if for_X==True:
                    if any(str(X[i][j]) in t for t in delete_list):
                        del Y[i][j]
                        del X[i][j]
                    else:
                        j+=1
                else:
                    if any(str(Y[i][j]) in t for t in delete_list):
                        del Y[i][j]
                        del X[i][j]
                    else:
                        j+=1
    return X, Y

### PIKLE LOAD & STORE
def dump(data, name):
    filehandler = open(name,"wb")
    pickle.dump(data, filehandler)
    filehandler.close()
def load(name):
    return pickle.load(open(name,'rb'))

def unfold_data(x_data, y_data):
	unfolded_x_data = []
	for sentence in x_data:
		for token in sentence:
			unfolded_x_data.append(token)
	unfolded_y_data = []
	for sentence in y_data:
		for token in sentence:
			unfolded_y_data.append(token)
	return unfolded_x_data, unfolded_y_data

def choice_feature_set_v2(type1, type2, type3, type4, temp_list):
    x1 = temp_list[0]
    x2 = temp_list[2]
    x3 = temp_list[4]
    x4 = temp_list[6]
    t1 = temp_list[1]
    t2 = temp_list[3]
    t3 = temp_list[5]
    t4 = temp_list[7]
    X_train = []
    for sentence in x4:
        temp1 = []
        for token in sentence:
            temp1.append([])
        X_train.append(temp1)
    X_test = []
    for sentence in t4:
        temp2 = []
        for token in sentence:
            temp2.append([])
        X_test.append(temp2)
    for i, sentence in enumerate(X_train):
        for j, token in enumerate(sentence):
            if type1 == 1:
                X_train[i][j] += x1[i][j]
            if type2 == 1:
                X_train[i][j] += x2[i][j]
            if type3 == 1:
                X_train[i][j] += x3[i][j]
            if type4 == 1:
                X_train[i][j] += x4[i][j]
    for i, sentence in enumerate(X_test):
        for j, token in enumerate(sentence):
            if type1 == 1:
                X_test[i][j] += t1[i][j]
            if type2 == 1:
                X_test[i][j] += t2[i][j]
            if type3 == 1:
                X_test[i][j] += t3[i][j]
            if type4 == 1:
                X_test[i][j] += t4[i][j]
    return X_train, X_test

def choice_feature_set(type1, type2, type3, type4, x1, x2, x3, x4, t1, t2, t3, t4):
    #assert( len(x1) == len(x2) == len(x3) == len(x4))
    #assert( len(t1) == len(t2) == len(t3) == len(t4))
    X_train = []
    for sentence in x4:
        temp1 = []
        for token in sentence:
            temp1.append([])
        X_train.append(temp1)
    X_test = []
    for sentence in t4:
        temp2 = []
        for token in sentence:
            temp2.append([])
        X_test.append(temp2)
    for i, sentence in enumerate(X_train):
        for j, token in enumerate(sentence):
            if type1 == 1:
                X_train[i][j] += x1[i][j]
            if type2 == 1:
                X_train[i][j] += x2[i][j]
            if type3 == 1:
                X_train[i][j] += x3[i][j]
            if type4 == 1:
                X_train[i][j] += x4[i][j]
    for i, sentence in enumerate(X_test):
        for j, token in enumerate(sentence):
            if type1 == 1:
                X_test[i][j] += t1[i][j]
            if type2 == 1:
                X_test[i][j] += t2[i][j]
            if type3 == 1:
                X_test[i][j] += t3[i][j]
            if type4 == 1:
                X_test[i][j] += t4[i][j]
    return X_train, X_test
