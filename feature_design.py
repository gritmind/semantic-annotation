from data_handler import dump, load
import collections
import operator
import nltk
from nltk import pos_tag
from nltk.corpus import treebank
from chunkers import ClassifierChunker
from nltk.corpus import treebank_chunk
from chunkers import TagChunker
from nltk.tokenize.moses import MosesDetokenizer
detokenizer = MosesDetokenizer()

import spacy
nlp = spacy.load('en')

from configparser import ConfigParser
##### ConfigParser
####################################
config = ConfigParser()
config.read('parameters.ini')
parser_jar_path = config.get('2_feature_extraction', 'parser_jar_path')
model_jar_path = config.get('2_feature_extraction', 'model_jar_path')
model_path = config.get('2_feature_extraction', 'model_path')

import os
#os.environ['STANFORD_PARSER'] = 'C:\\stanford-parser-full-2016-10-31\\stanford-parser.jar'
#os.environ['STANFORD_MODELS'] = 'C:\\stanford-parser-full-2016-10-31\\stanford-parser-3.7.0-models.jar'
os.environ['STANFORD_PARSER'] = parser_jar_path
os.environ['STANFORD_MODELS'] = model_jar_path
from nltk.parse.stanford import StanfordParser
#parser = StanfordParser(model_path='edu\\stanford\\nlp\\models\\lexparser\\englishPCFG.ser.gz')
parser = StanfordParser(model_path=model_path)


#####################################################
""" spaCy (dependency parser) feature extraction """
#####################################################

def create_lookup_spacy_pos():
    list_spacy_pos = ['DET', 'ADJ', 'NOUN', 'VERB', 'PUNCT', 'CONJ', 'ADP', 'PART', 'ADV', 'PRON', 'NUM', 'PROPN', 'INTJ', 'X', 'SYM']
    lookup_table_spacy_pos = { }
    for i, token in enumerate(list_spacy_pos):
        lookup_table_spacy_pos[token] = i # POS에서는 1부터(i+1) 시작안해도 된다. POS에는 unknown이 없기 때문이다.
    print('>> spacy pos lookup table dim = ', len(lookup_table_spacy_pos))
    return lookup_table_spacy_pos

def create_lookup_spacy_dependency():
    ### For dependent look-up table
    list_dependent = [
        'det','amod','compound','nsubj','aux','ROOT','dobj','punct','appos','conj','cc','prep','pobj','pcomp','nsubjpass',\
        'auxpass','xcomp','advmod','advcl','acl','neg','relcl','poss','mark','ccomp','nummod','agent','acomp','expl','attr',\
        'intj','oprd','preconj','nmod','npadvmod','prt','csubj','quantmod','case','predet','dep','dative','parataxis']
    lookup_table_dependent = { }
    for i, token in enumerate(list_dependent):
        lookup_table_dependent[token] = i
    print('>> spacy dependency lookup table dim = ', len(lookup_table_dependent))
    return lookup_table_dependent

def add_spacy_pos(FE4_X_train, FE4_X_test, Xtrain, Xtest, lookup_spacy_pos, window_size_spacy_pos):
    before = len(FE4_X_train[0][0])
    ## train set
    for i, sentence in enumerate(Xtrain):
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)
        ### for exception [START]
        list_detokenized_sent = list(detokenized_sent)
        if list_detokenized_sent[-1] == '.' and list_detokenized_sent[-2].isupper() == True:
            list_detokenized_sent[-1] = ' .'
        new_detokenized_sent = "".join(list_detokenized_sent)
        ### for exception [END]
        spacy_sent = nlp(new_detokenized_sent)

        #print(len(sentence), len(spacy_sent))
        #print(sentence)
        #print(spacy_sent)
        assert(len(sentence)==len(spacy_sent))
        ##################################################################################################
        for j, token in enumerate(spacy_sent):
            FE4_X_train[i][j] += FE_spaCy_POS(j, spacy_sent, lookup_spacy_pos, window_size_spacy_pos)
    ## test set
    for i, sentence in enumerate(Xtest):
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)
        ### for exception [START]
        list_detokenized_sent = list(detokenized_sent)
        if list_detokenized_sent[-1] == '.' and list_detokenized_sent[-2].isupper() == True:
            list_detokenized_sent[-1] = ' .'
        new_detokenized_sent = "".join(list_detokenized_sent)
        ### for exception [END]
        spacy_sent = nlp(new_detokenized_sent)
        assert(len(sentence)==len(spacy_sent))
        ##################################################################################################
        for j, token in enumerate(spacy_sent):
            FE4_X_test[i][j] += FE_spaCy_POS(j, spacy_sent, lookup_spacy_pos, window_size_spacy_pos)
    after = len(FE4_X_train[0][0])
    print('>> spacy pos feature vector dim = ', after-before)
    return FE4_X_train, FE4_X_test

def add_spacy_chunk(FE4_X_train, FE4_X_test, Xtrain, Xtest):
    before = len(FE4_X_train[0][0])

    ## training
    for i, sentence in enumerate(Xtrain):
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)
        ### for exception [START]
        list_detokenized_sent = list(detokenized_sent)
        if list_detokenized_sent[-1] == '.' and list_detokenized_sent[-2].isupper() == True:
            list_detokenized_sent[-1] = ' .'
        new_detokenized_sent = "".join(list_detokenized_sent)
        ### for exception [END]
        spacy_sent = nlp(new_detokenized_sent)
        assert(len(sentence)==len(spacy_sent))
        npchunk_numbered_sent = numbering_npchunk(spacy_sent)
        ppchunk_numbered_sent = numbering_ppchunk(npchunk_numbered_sent, spacy_sent)
        pp_isjustright_fromnp_numbered_sent = numbering_pp_isJustRight_fromNP_sent(npchunk_numbered_sent, ppchunk_numbered_sent)
        ### For just right from Root (밑에 3개는 0또는1밖에없다.)
        firstNP_fromroot_sent = Xchunk_isFirstRight_fromRoot(npchunk_numbered_sent, spacy_sent)
        firstPP_fromroot_sent = Xchunk_isFirstRight_fromRoot(ppchunk_numbered_sent, spacy_sent)
        firstPPjustrightNP_fromroot_sent = Xchunk_isFirstRight_fromRoot(pp_isjustright_fromnp_numbered_sent, spacy_sent)
        ## for what is the prep
        whichprep_sent = numbering_which_ppchunk(ppchunk_numbered_sent, spacy_sent)

        ####################
        for j, token in enumerate(spacy_sent):
            FE4_X_train[i][j] += is_first_np_chunk(j, npchunk_numbered_sent)
            FE4_X_train[i][j] += is_last_np_chunk(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += is_np_chuck(j, spacy_sent)
            FE4_X_train[i][j] += np_isLeft_fromRoot(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += np_isRight_fromRoot(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += Xchunk_isLeft_fromSubRoot(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += Xchunk_isRight_fromSubRoot(j, npchunk_numbered_sent, spacy_sent)
            # ppchunk_numbered_sent
            FE4_X_train[i][j] += is_Xchunk(j, ppchunk_numbered_sent)
            FE4_X_train[i][j] += Xchunk_isLeft_fromRoot(j, ppchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += Xchunk_isRight_fromRoot(j, ppchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += Xchunk_isLeft_fromSubRoot(j, ppchunk_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += Xchunk_isRight_fromSubRoot(j, ppchunk_numbered_sent, spacy_sent)
            # pp_isjustright_fromnp_numbered_sent
            FE4_X_train[i][j] += is_Xchunk(j, pp_isjustright_fromnp_numbered_sent)
            FE4_X_train[i][j] += Xchunk_isLeft_fromRoot(j, pp_isjustright_fromnp_numbered_sent, spacy_sent)
            FE4_X_train[i][j] += Xchunk_isRight_fromRoot(j, pp_isjustright_fromnp_numbered_sent, spacy_sent)
            # For just right from Root
            FE4_X_train[i][j] += is_Xchunk(j, firstNP_fromroot_sent)
            FE4_X_train[i][j] += is_Xchunk(j, firstPP_fromroot_sent)
            FE4_X_train[i][j] += is_Xchunk(j, firstPPjustrightNP_fromroot_sent)
            # for what is the prep
            FE4_X_train[i][j] += whichprep_sent[j]
    ## test set
    for i, sentence in enumerate(Xtest):
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)
        ### for exception [START]
        list_detokenized_sent = list(detokenized_sent)
        if list_detokenized_sent[-1] == '.' and list_detokenized_sent[-2].isupper() == True:
            list_detokenized_sent[-1] = ' .'
        new_detokenized_sent = "".join(list_detokenized_sent)
        ### for exception [END]
        spacy_sent = nlp(new_detokenized_sent)
        assert(len(sentence)==len(spacy_sent))
        npchunk_numbered_sent = numbering_npchunk(spacy_sent)
        ppchunk_numbered_sent = numbering_ppchunk(npchunk_numbered_sent, spacy_sent)
        pp_isjustright_fromnp_numbered_sent = numbering_pp_isJustRight_fromNP_sent(npchunk_numbered_sent, ppchunk_numbered_sent)
        assert(len(npchunk_numbered_sent)==len(ppchunk_numbered_sent)==len(pp_isjustright_fromnp_numbered_sent))
        ### For just right from Root
        firstNP_fromroot_sent = Xchunk_isFirstRight_fromRoot(npchunk_numbered_sent, spacy_sent)
        firstPP_fromroot_sent = Xchunk_isFirstRight_fromRoot(ppchunk_numbered_sent, spacy_sent)
        firstPPjustrightNP_fromroot_sent = Xchunk_isFirstRight_fromRoot(pp_isjustright_fromnp_numbered_sent, spacy_sent)
        ## for what is the prep
        whichprep_sent = numbering_which_ppchunk(ppchunk_numbered_sent, spacy_sent)
        ####################
        for j, token in enumerate(spacy_sent):
            FE4_X_test[i][j] += is_first_np_chunk(j, npchunk_numbered_sent)
            FE4_X_test[i][j] += is_last_np_chunk(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += is_np_chuck(j, spacy_sent)
            FE4_X_test[i][j] += np_isLeft_fromRoot(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += np_isRight_fromRoot(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += Xchunk_isLeft_fromSubRoot(j, npchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += Xchunk_isRight_fromSubRoot(j, npchunk_numbered_sent, spacy_sent)
            # ppchunk_numbered_sent
            FE4_X_test[i][j] += is_Xchunk(j, ppchunk_numbered_sent)
            FE4_X_test[i][j] += Xchunk_isLeft_fromRoot(j, ppchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += Xchunk_isRight_fromRoot(j, ppchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += Xchunk_isLeft_fromSubRoot(j, ppchunk_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += Xchunk_isRight_fromSubRoot(j, ppchunk_numbered_sent, spacy_sent)
            # pp_isjustright_fromnp_numbered_sent
            FE4_X_test[i][j] += is_Xchunk(j, pp_isjustright_fromnp_numbered_sent)
            FE4_X_test[i][j] += Xchunk_isLeft_fromRoot(j, pp_isjustright_fromnp_numbered_sent, spacy_sent)
            FE4_X_test[i][j] += Xchunk_isRight_fromRoot(j, pp_isjustright_fromnp_numbered_sent, spacy_sent)
            # For just right from Root
            FE4_X_test[i][j] += is_Xchunk(j, firstNP_fromroot_sent)
            FE4_X_test[i][j] += is_Xchunk(j, firstPP_fromroot_sent)
            FE4_X_test[i][j] += is_Xchunk(j, firstPPjustrightNP_fromroot_sent)
            # for what is the prep
            FE4_X_test[i][j] += whichprep_sent[j]
    after = len(FE4_X_train[0][0])
    # [A clinical lab section, the clinical site name, day, time, the lab]
    # A clinical lab section shall include the clinical site name, the class, instructor, day and time of the lab.
    # [1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 5, 5, 0]
    print('>> spacy chunk feature vector dim = ', after-before)
    return FE4_X_train, FE4_X_test

def add_spacy_parser(FE4_X_train, FE4_X_test, Xtrain, Xtest, lookup_spacy_pos, lookup_spacy_dependency):
    before = len(FE4_X_train[0][0])

    ### train set
    for i, sentence in enumerate(Xtrain):
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)
        ### for exception [START]
        list_detokenized_sent = list(detokenized_sent)
        if list_detokenized_sent[-1] == '.' and list_detokenized_sent[-2].isupper() == True:
            list_detokenized_sent[-1] = ' .'
        new_detokenized_sent = "".join(list_detokenized_sent)
        ### for exception [END]
        spacy_sent = nlp(new_detokenized_sent)
        assert(len(sentence)==len(spacy_sent))
        # For Normalization
        len_sentence = len(sentence)
        max_len_heads = lenMax_amongHeadList_toRoot(spacy_sent)
        max_len_subtrees = lenMax_amongSubtreeList_toRoot(spacy_sent)

        for j, token in enumerate(spacy_sent):
            # Distance
            FE4_X_train[i][j] += normalization(len_sentence, 1, j+1)
            FE4_X_train[i][j] += normalization(max_len_heads, 0, len(headList_to_root(token)))
            # Current token
            FE4_X_train[i][j] += normalization(max_len_subtrees, 1, len(list(token.subtree)))
            FE4_X_train[i][j] += what_is_myDependent(token, lookup_spacy_dependency)
            FE4_X_train[i][j] += is_LeftRight_fromRoot(j, spacy_sent)
            # Head token
            FE4_X_train[i][j] += normalization(max_len_subtrees, 1, len(list(token.head.subtree)))
            FE4_X_train[i][j] += what_is_headDependent(token, lookup_spacy_dependency)
            FE4_X_train[i][j] += what_is_headPOS(token, lookup_spacy_pos)
            # Implicitly Grouping
            headlist_to_root = [(dependent_token.dep_) for dependent_token in headList_to_root(token)]
            headpath_to_root = remove_duplicate(headlist_to_root)
            FE4_X_train[i][j] += high_level_implicit_grouping(headpath_to_root)
            # for sub prioirty
            FE4_X_train[i][j] += is_subpriority(j, spacy_sent)
            # SUB ROOT & SUB NOUN
            FE4_X_train[i][j] += is_middle_between_RtAndSubRt(j, spacy_sent)
            FE4_X_train[i][j] += is_Right_fromSubRoot(j, spacy_sent)
            ### Ancestors와 children은 패턴이 안보여서 일단 pass
            #print(token, '---' , headList_to_root(token))
            #print(token, len(list(token.subtree)))
            #print(token, '\t', len(list(token.subtree)),len(list(token.ancestors)),len(list(token.children)), '\t\t', list(token.ancestors), '\t\t', list(token.children)  )
    #     for j, token in enumerate(spacy_sent):
    #         print(token.orth_,'\t', token.dep_,'\t', token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])

    #     print('-----------------------------------------------------------------------------------------------------')

    ### test set
    for i, sentence in enumerate(Xtest):
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)
        ### for exception [START]
        list_detokenized_sent = list(detokenized_sent)
        if list_detokenized_sent[-1] == '.' and list_detokenized_sent[-2].isupper() == True:
            list_detokenized_sent[-1] = ' .'
        new_detokenized_sent = "".join(list_detokenized_sent)
        ### for exception [END]
        spacy_sent = nlp(new_detokenized_sent)
        assert(len(sentence)==len(spacy_sent))
        # For Normalization
        len_sentence = len(sentence)
        max_len_heads = lenMax_amongHeadList_toRoot(spacy_sent)
        max_len_subtrees = lenMax_amongSubtreeList_toRoot(spacy_sent)

        for j, token in enumerate(spacy_sent):
            # Distance
            FE4_X_test[i][j] += normalization(len_sentence, 1, j+1)
            FE4_X_test[i][j] += normalization(max_len_heads, 0, len(headList_to_root(token)))
            # Current token
            FE4_X_test[i][j] += normalization(max_len_subtrees, 1, len(list(token.subtree)))
            FE4_X_test[i][j] += what_is_myDependent(token, lookup_spacy_dependency)
            FE4_X_test[i][j] += is_LeftRight_fromRoot(j, spacy_sent)
            # Head token
            FE4_X_test[i][j] += normalization(max_len_subtrees, 1, len(list(token.head.subtree)))
            FE4_X_test[i][j] += what_is_headDependent(token, lookup_spacy_dependency)
            FE4_X_test[i][j] += what_is_headPOS(token, lookup_spacy_pos)
            # Implicitly Grouping
            headlist_to_root = [(dependent_token.dep_) for dependent_token in headList_to_root(token)]
            headpath_to_root = remove_duplicate(headlist_to_root)
            FE4_X_test[i][j] += high_level_implicit_grouping(headpath_to_root)
            # for sub prioirty
            FE4_X_test[i][j] += is_subpriority(j, spacy_sent)
            # SUB ROOT & SUB NOUN
            FE4_X_test[i][j] += is_middle_between_RtAndSubRt(j, spacy_sent)
            FE4_X_test[i][j] += is_Right_fromSubRoot(j, spacy_sent)
            ### Ancestors와 children은 패턴이 안보여서 일단 pass
            #print(token, '---' , headList_to_root(token))
            #print(token, len(list(token.subtree)))
            #print(token, '\t', len(list(token.subtree)),len(list(token.ancestors)),len(list(token.children)), '\t\t', list(token.ancestors), '\t\t', list(token.children)  )
    #     for j, token in enumerate(spacy_sent):
    #         print(token.orth_,'\t', token.dep_,'\t', token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])

    #     print('-----------------------------------------------------------------------------------------------------')

    after = len(FE4_X_train[0][0])
    print('>> spacy dependency parser feature vector dim = ', after-before)
    return FE4_X_train, FE4_X_test




#####################################################
""" spaCy (dependency parser) feature design """
#####################################################

import nltk
import collections
import operator


def idx_Root(spacy_sent):
    verb_withNsubtree = {}
    for i, token in enumerate(spacy_sent):
        if token.pos_ == 'VERB':
            verb_withNsubtree[i] = len(list(token.subtree))
    verb_withNsubtree = sorted(verb_withNsubtree.items(), key=operator.itemgetter(1), reverse=True)

    try:
        return verb_withNsubtree[0][0] # index for root
    except IndexError:
        for i, token in enumerate(spacy_sent):
            if token.dep_ == 'ROOT':
                idx_root = i
        return idx_root



def idx_SubRoot(spacy_sent):
    no_verb = True
    verb_withNsubtree = {}
    for i, token in enumerate(spacy_sent):
        if token.pos_ == 'VERB':
            verb_withNsubtree[i] = len(list(token.subtree))
            no_verb = False
    verb_withNsubtree = sorted(verb_withNsubtree.items(), key=operator.itemgetter(1), reverse=True)


    if no_verb == True: # 만약에 verb가 아예 없으면 그냥 root idx내보냄
        for i, token in enumerate(spacy_sent):
            if token.dep_ == 'ROOT':
                idx_root = i
        return idx_root
    else:
        try:
            return verb_withNsubtree[1][0]
        except IndexError:
            return verb_withNsubtree[0][0] # verb가 1개만 있으면 그냥 root idx내보냄



##### spacy pos

def what_is_headPOS(token, lookup):

    token_feature = [0] * len(lookup)
    token_feature[lookup[token.head.pos_]] = 1
    return token_feature


def spacy_vectorized_token_using_pos_lookup(j_plus_alpha, spacy_sent, lookup):

    # token feature 0으로 초기화
    token_feature = [0] * len(lookup)

    ### 1. for array index exception
    if j_plus_alpha < 0: # for first index exception
        return token_feature

    try:
        spacy_sent[j_plus_alpha] # f.;/or last index exception
    except IndexError:
        return token_feature # 모든 element가 0인 token feature

    ### 2. for key (unknown token) exception
    # in pos, there is no exception for lookup table
#     try:
#         lookup[sentence[j_plus_alpha]]
#     except KeyError:
#         if unknown == True:
#             token_feature[lookup['unknown_token']-1] = 1
#             return token_feature # which is unknown index
#         else: # unknown token 적용안할 것이다...
#             return token_feature # 따라서, 그냥 0벡터로..

    ### 3. oridinal case
#     token_feature[lookup[sentence[j_plus_alpha]]-1] = 1 # unknown pos가 없으니 -1은 하지 않는다.
    token_feature[lookup[spacy_sent[j_plus_alpha].pos_]] = 1
    return token_feature


def FE_spaCy_POS(j, spacy_sent, lookup_table_POS, window_size_pos): # Feature extraction using POS

    if window_size_pos == 1:
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        return t__0
    elif window_size_pos == 3:
        t_m1 = spacy_vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = spacy_vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        return t_m1 + t__0 + t_p1
    elif window_size_pos == 5:
        t_m2 = spacy_vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = spacy_vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = spacy_vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = spacy_vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        return t_m2 + t_m1 + t__0 + t_p1 + t_p2
    elif window_size_pos == 7:
        t_m3 = spacy_vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = spacy_vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = spacy_vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = spacy_vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = spacy_vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = spacy_vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        return t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3
    elif window_size_pos == 9:
        t_m4 = spacy_vectorized_token_using_pos_lookup(j-4, spacy_sent, lookup_table_POS)
        t_m3 = spacy_vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = spacy_vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = spacy_vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = spacy_vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = spacy_vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = spacy_vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        t_p4 = spacy_vectorized_token_using_pos_lookup(j+4, spacy_sent, lookup_table_POS)
        return t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4
    elif window_size_pos == 11:
        t_m5 = spacy_vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
        t_m4 = spacy_vectorized_token_using_pos_lookup(j-4, spacy_sent, lookup_table_POS)
        t_m3 = spacy_vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = spacy_vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = spacy_vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = spacy_vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = spacy_vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = spacy_vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        t_p4 = spacy_vectorized_token_using_pos_lookup(j+4, spacy_sent, lookup_table_POS)
        t_p5 = spacy_vectorized_token_using_pos_lookup(j+5, spacy_sent, lookup_table_POS)
        return t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5
    elif window_size_pos == 13:
        t_m6 = spacy_vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = spacy_vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
        t_m4 = spacy_vectorized_token_using_pos_lookup(j-4, spacy_sent, lookup_table_POS)
        t_m3 = spacy_vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = spacy_vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = spacy_vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = spacy_vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = spacy_vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = spacy_vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = spacy_vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        t_p4 = spacy_vectorized_token_using_pos_lookup(j+4, spacy_sent, lookup_table_POS)
        t_p5 = spacy_vectorized_token_using_pos_lookup(j+5, spacy_sent, lookup_table_POS)
        t_p6 = spacy_vectorized_token_using_pos_lookup(j+6, spacy_sent, lookup_table_POS)
        return t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6
    elif window_size_pos == 15:
        t_m7 = spacy_vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = spacy_vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = spacy_vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
        t_m4 = vectorized_token_using_pos_lookup(j-4, spacy_sent, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, spacy_sent, lookup_table_POS)
        t_p5 = vectorized_token_using_pos_lookup(j+5, spacy_sent, lookup_table_POS)
        t_p6 = vectorized_token_using_pos_lookup(j+6, spacy_sent, lookup_table_POS)
        t_p7 = vectorized_token_using_pos_lookup(j+7, spacy_sent, lookup_table_POS)
        return t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7


##################################
""" Chunking (Shallow Parsing) """
##################################

def numbering_which_ppchunk(ppchunk_numbered_sent, spacy_sent):

    # make prep lookup table
    prep_list = ['for','to','into','with','upon','of','as','by','on','in','unknown']
    prep_lookup_table = make_lookup_table(prep_list)


    # init feature vector per token
    feature_2d = []
    whichprep_sent = [0] * len(ppchunk_numbered_sent)


    for i, num_ in enumerate(ppchunk_numbered_sent):

        # first token
        if i == 0 and ppchunk_numbered_sent[i+1] !=0:
            if str(spacy_sent[i]) in prep_list:
                whichprep_sent[i] = prep_lookup_table[str(spacy_sent[i])]+1
            else:
                whichprep_sent[i] = prep_lookup_table['unknown']+1

        # not first token
        if i !=0 and num_ !=0 and ppchunk_numbered_sent[i] != ppchunk_numbered_sent[i-1]: # 전치사위치: 배열의 마지막 token이 아니고, 숫자0이 아니고, 전(t-1) token과 현재(t) token이 다를 때..


            if str(spacy_sent[i]) in prep_list:
                whichprep_sent[i] = prep_lookup_table[str(spacy_sent[i])]+1
            else:
                whichprep_sent[i] = prep_lookup_table['unknown']+1

            #try:
            #    whichprep_sent[i] = prep_lookup_table[spacy_sent[i]]+1
            #except KeyError:
            #    whichprep_sent[i] = prep_lookup_table['unknown']+1

            #whichprep_sent[i] = prep_lookup_table[spacy_sent[i]]

    for i, num_ in enumerate(whichprep_sent):

        if num_ != 0:

            for j in range(i, len(whichprep_sent)):
                if not j == len(whichprep_sent)-1:
                    if ppchunk_numbered_sent[j] == ppchunk_numbered_sent[j+1]:
                        whichprep_sent[j] = num_
                        whichprep_sent[j+1] = num_
                    else:
                        break
    # make 2d array...
    for i, num_ in enumerate(whichprep_sent):
        whichprep_feature = [0] * len(prep_lookup_table)
        if num_ == 0:
            feature_2d.append(whichprep_feature)
        else:
            idx = num_ - 1
            whichprep_feature[idx] = 1
            feature_2d.append(whichprep_feature)

    return feature_2d

def is_Xchunk(j, numbered_sent):

    if numbered_sent[j] != 0:
        return [1]
    else:
        return [0]

def Xchunk_isFirstRight_fromRoot(numbered_sent, spacy_sent):

    isFirstRight_fromRoot_sent = [0] * len(spacy_sent)

    # find root word
    for i, token in enumerate(spacy_sent):
        if token.dep_ == 'ROOT':
            idx_root = i

    # who is the first right chunk from root
    for i, num in enumerate(numbered_sent):
        if numbered_sent[i] != 0 and i > idx_root:
            isFirstRight_fromRoot_sent[i] = 1
            for k, num in enumerate(numbered_sent):
                if numbered_sent[k] == numbered_sent[i]:
                    isFirstRight_fromRoot_sent[k] = 1
            break # 첫 번째만 필요하므로 그 다음은 break해서 빠져나온다.

    return isFirstRight_fromRoot_sent

###################################################################################### FOR ROOT

def Xchunk_isLeft_fromRoot(j, numbered_sent, spacy_sent):

    # find root word
    for i, token in enumerate(spacy_sent):
        if token.dep_ == 'ROOT':
            idx_root = i

    if j < idx_root and numbered_sent[j] != 0: # 물론 pp인 상태에서 왼쪽에 있어야 한다.
        return [1] # left from root
    else:
        return [0] # right from root

def Xchunk_isRight_fromRoot(j, numbered_sent, spacy_sent):

    # find root word
    for i, token in enumerate(spacy_sent):
        if token.dep_ == 'ROOT':
            idx_root = i

    if j > idx_root and numbered_sent[j] != 0: # 물론 pp인 상태에서 왼쪽에 있어야 한다.
        return [1] # left from root
    else:
        return [0] # right from root

###################################################################################### FOR SUB ROOT

def Xchunk_isLeft_fromSubRoot(j, numbered_sent, spacy_sent):

    idx_SubRt = idx_SubRoot(spacy_sent)

    if j < idx_SubRt and numbered_sent[j] != 0: # 물론 pp인 상태에서 왼쪽에 있어야 한다.
        return [1] # left from root
    else:
        return [0] # right from root

def Xchunk_isRight_fromSubRoot(j, numbered_sent, spacy_sent):

    idx_SubRt = idx_SubRoot(spacy_sent)

    if j > idx_SubRt and numbered_sent[j] != 0: # 물론 pp인 상태에서 왼쪽에 있어야 한다.
        return [1] # left from root
    else:
        return [0] # right from root

def numbering_ppchunk(npchunk_numbered_sent, spacy_sent):

    ppchunk_numbered_sent = [0] * len(spacy_sent)
    idx_prep = 1

    # Find PP pattern
    for i, num_or_token in enumerate(npchunk_numbered_sent):

        if not i == len(npchunk_numbered_sent)-1: # Boundary Exception: not last token
            if npchunk_numbered_sent[i] == 0 and npchunk_numbered_sent[i+1] != 0:
                if spacy_sent[i].pos_ == 'ADP': # if preposition
                    ppchunk_numbered_sent[i] = idx_prep
                    for k, num in enumerate(npchunk_numbered_sent):
                        if npchunk_numbered_sent[k] == npchunk_numbered_sent[i+1]:
                            ppchunk_numbered_sent[k] = idx_prep
                    idx_prep += 1


    return ppchunk_numbered_sent

def numbering_pp_isJustRight_fromNP_sent(npchunk_numbered_sent, ppchunk_numbered_sent): # NP - PP

    pp_isjustright_fromnp = [0] * len(npchunk_numbered_sent)
    idx_prep = 1

    # Make pp numbered list (which is just right from np)
    for i, num in enumerate(ppchunk_numbered_sent):
        if not i == 0: # Boundary Exception: not last token
            if ppchunk_numbered_sent[i-1] != ppchunk_numbered_sent[i] and ppchunk_numbered_sent[i] != 0: # 앞에와 번호가 다르고, 현재가 pp이어야만 한다.
                if npchunk_numbered_sent[i-1] != 0 and npchunk_numbered_sent[i] == 0: # 바로 전이 np이다. 즉, np right position에 있다.
                    pp_isjustright_fromnp[i] = idx_prep
                    for k, num in enumerate(ppchunk_numbered_sent):
                        if ppchunk_numbered_sent[k] == ppchunk_numbered_sent[i]:
                            pp_isjustright_fromnp[k] = idx_prep
                    idx_prep += 1

    return pp_isjustright_fromnp

def is_np_chuck(j, spacy_sent):
    list_np_word = []
    list_np_chunk = list(spacy_sent.noun_chunks)
    for chunk in list_np_chunk:
        for token in chunk:
            list_np_word.append(token)

    if spacy_sent[j] in list_np_word:
        return [1]
    else:
        return [0]

def numbering_npchunk(spacy_sent):

    list_np_chunk = list(spacy_sent.noun_chunks)
    check_listnpchunk = []
    for chunk in list_np_chunk:
        temp_list = [-1] * len(str(chunk).split())
        check_listnpchunk.append(temp_list)
    check_spacysent = [0] * len(spacy_sent)

    ###
    for i, token in enumerate(spacy_sent):
        #
        for m, chunk in enumerate(list_np_chunk):
            token_list = str(chunk).split()
            for n, token in enumerate(token_list):

                if check_listnpchunk[m][n] == -1: # condition: 반영안된 것만 유효
                    if spacy_sent[i] == chunk[n]:
                        check_spacysent[i] = m + 1 # index가 0부터 시작하므로..
                        check_listnpchunk[m][n] = 0 # -1 -> 0으로.. (반영한다는 뜻)
                        break # check_spacysent[i]가 단 한번만 할당될 수 있도록 한 번되면 break하자. 안그럼 더 돌면서 overlap된다.

    chunk_numbered_sent = check_spacysent
    return chunk_numbered_sent # [1,1,1, 0,0, 2,2, 0,0,0, 3,3,3]


def is_first_np_chunk(j, chunk_numbered_sent):

    if chunk_numbered_sent[j] == 1:
        return [1]
    else:
        return [0]

def is_last_np_chunk(j, chunk_numbered_sent, spacy_sent):
    list_np_chunk = list(spacy_sent.noun_chunks)

    if chunk_numbered_sent[j] == len(list_np_chunk):
        return [1]
    else:
        return [0]



def np_isLeft_fromRoot(j, chunk_numbered_sent, spacy_sent):

    # find root word
    for i, token in enumerate(spacy_sent):
        if token.dep_ == 'ROOT':
            idx_root = i

    if j < idx_root and chunk_numbered_sent[j] != 0: # 물론 np인 상태에서 왼쪽에 있어야 한다.
        return [1] # left from root
    else:
        return [0] # right from root

def np_isRight_fromRoot(j, chunk_numbered_sent, spacy_sent):

    # find root word
    for i, token in enumerate(spacy_sent):
        if token.dep_ == 'ROOT':
            idx_root = i

    if j > idx_root and chunk_numbered_sent[j] != 0: # # 물론 np인 상태에서 왼쪽에 있어야 한다.
        return [1] # np인 동시에 오른쪽
    else:
        return [0]


##### dependency parsing

def is_middle_between_RtAndSubRt(j, spacy_sent):
    idx_Rt = idx_Root(spacy_sent)
    idx_SubRt = idx_SubRoot(spacy_sent)
    if j > idx_Rt and j < idx_SubRt:
        return [1]
    else:
        return [0]

def is_Right_fromSubRoot(j, spacy_sent):
    idx_SubRt = idx_SubRoot(spacy_sent)
    if j > idx_SubRt:
        return [1]
    else:
        return [0]

def is_subpriority(j, spacy_sent):
    if spacy_sent[j].dep_ == 'aux':
        if spacy_sent[j].head.dep_ == 'acl' or spacy_sent[j].head.dep_ == 'relcl' or spacy_sent[j].head.dep_ == 'advcl':
            return [1]
        if spacy_sent[j].head.dep_ == 'conj' and spacy_sent[j].head.head.dep_ == 'ROOT':
            return [1]
    return [0]

#def is_aux_to_ROOT(j, spacy_sent): # 오로지 sub prioirty만을 위한 feature
#    if spacy_sent[j].dep_ == 'aux':
#        if spacy_sent[j].head.dep_ == 'ccomp':
#            return [1]
#    return [0]

def make_lookup_table(list_):
    lookup_table = {}
    for i, token in enumerate(list_):
        lookup_table[token] = i
    return lookup_table

####################################################################################################
##### FOR IMPLICIT GROUPING (START)
def remove_duplicate(headList_to_root):
    new_list = []
    #temp_list = headList_to_root[:] # <--- copy only value
    #temp_list = headList_to_root # <--- copy value with reference
    for i, dep_ in enumerate(headList_to_root):
        if not i == len(headList_to_root)-1:
            if not headList_to_root[i] == headList_to_root[i+1]:
                new_list.append(headList_to_root[i])
        else:
            new_list.append(headList_to_root[i])
    return new_list

def vectorized_dep_feature(dep_, lookup_table):
    dep_feature = [0] * len(lookup_table)

    try:
        lookup_table[dep_]
    except KeyError:
        return dep_feature # 그냥 0벡터로..

    dep_feature[lookup_table[dep_]] = 1
    return dep_feature

def high_level_implicit_grouping(headpath_to_root):

    # Dependency Implicitly Grouping Dictionary
    ### For dependent look-up table
    list_acl = ['dobj', 'nsubj', 'xcomp', 'pobj', 'ROOT', 'attr', 'nsubjpass', 'conj']
    list_relcl = ['dobj', 'pobj', 'npadvmod', 'attr', 'ROOT', 'conj', 'dep', 'nsubjpass', 'nsubj']
    list_prep = ['ROOT','dobj','pobj','conj','advmod','pcomp','xcomp','nsubj','ccomp','nsubjpass','punct','appos','amod','acomp','attr','npadvmod','dep']
    list_dobj = ['acomp', 'ccomp', 'xcomp', 'ROOT']
    list_pobj = ['acomp', 'ccomp', 'xcomp', 'ROOT']
    list_nsubj  = ['acomp', 'ccomp', 'xcomp', 'ROOT']
    list_nsubjpass = ['acomp', 'ccomp', 'xcomp', 'ROOT']

    lookup_table_acl = make_lookup_table(list_acl)
    lookup_table_relcl = make_lookup_table(list_relcl)
    lookup_table_prep = make_lookup_table(list_prep)
    lookup_table_dobj = make_lookup_table(list_dobj)
    lookup_table_pobj = make_lookup_table(list_pobj)
    lookup_table_nsubj = make_lookup_table(list_nsubj)
    lookup_table_nsubjpass = make_lookup_table(list_nsubjpass)

    acl_Vec = [0] * len(lookup_table_acl)
    relcl_Vec = [0] * len(lookup_table_relcl)
    prep_Vec = [0] * len(lookup_table_prep)
    dobj_Vec = [0] * len(lookup_table_dobj)
    pobj_Vec = [0] * len(lookup_table_pobj)
    nsubj_Vec = [0] * len(lookup_table_nsubj)
    nsubjpass_Vec = [0] * len(lookup_table_nsubjpass)

    #acl_Vec = [0]
    #relcl_Vec = [0]
    #prep_Vec = [0]
    #dobj_Vec = [0]
    #pobj_Vec = [0]
    #nsubj_Vec = [0]
    #nsubjpass_Vec = [0]

    root_Vec = [0] # is ROOT (len()==0)
    advcl_Vec = [0] # is advcl

    # Step 1:
    if len(headpath_to_root) == 0:
        root_Vec[0] = 1
        yes_no = 1
    elif 'advcl' in headpath_to_root:
        advcl_Vec[0] = 1
        yes_no = 1
    else:
    # Step 2:
        if 'acl' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'acl':
                        #acl_Vec[0] = 1
                        acl_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_acl)
                        break
        elif 'relcl' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'relcl':
                        #relcl_Vec[0] = 1
                        relcl_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_relcl)
                        break
        elif 'prep' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'prep':
                        #prep_Vec[0] = 1
                        prep_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_prep)
                        break
        elif 'dobj' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'dobj':
                        #dobj_Vec[0] = 1
                        dobj_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_dobj)
                        break
        elif 'pobj' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'pobj':
                        #pobj_Vec[0] = 1
                        pobj_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_pobj)
                        break
        elif 'nsubj' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'nsubj':
                        #nsubj_Vec[0] = 1
                        nsubj_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_nsubj)
                        break
        elif 'nsubjpass' in headpath_to_root:
            for i, dep_ in enumerate(headpath_to_root):
                if not i == len(headpath_to_root)-1:
                    if headpath_to_root[i] == 'nsubjpass':
                        #nsubjpass_Vec[0] = 1
                        nsubjpass_Vec = vectorized_dep_feature(headpath_to_root[i+1], lookup_table_nsubjpass)
                        break

    return root_Vec + advcl_Vec + acl_Vec + relcl_Vec + prep_Vec + dobj_Vec + pobj_Vec + nsubj_Vec + nsubjpass_Vec


##### FOR IMPLICIT GROUPING (END)
####################################################################################################

def is_LeftRight_fromRoot(j_alpha, spacy_sent):
    # find root word
    for i, token in enumerate(spacy_sent):
        if token.dep_ == 'ROOT':
            idx_root = i
    if j_alpha < i:
        return [1]
    else:
        return [0]

def what_is_headPOS(token, lookup):
    token_feature = [0] * len(lookup)
    token_feature[lookup[token.head.pos_]] = 1
    return token_feature

def what_is_myDependent(token, lookup):
    token_feature = [0] * len(lookup)
    token_feature[lookup[token.head.dep_]] = 1
    return token_feature


def what_is_headDependent(token, lookup):

    token_feature = [0] * len(lookup)
    token_feature[lookup[token.head.dep_]] = 1
    return token_feature


def headList_to_root(token): # REFERENCE: https://github.com/NSchrading/intro-spacy-nlp
    # Write a function that walks up the syntactic tree of the given token and
    # collects all tokens to the root token (including root token).
    """
    Walk up the syntactic tree, collecting tokens to the root of the given `token`.
    :param token: Spacy token
    :return: list of Spacy tokens
    """
    tokens_to_r = []
    while token.head is not token:
        tokens_to_r.append(token)
        token = token.head
        tokens_to_r.append(token)
    return tokens_to_r

def lenMax_amongHeadList_toRoot(spacy_sent):
    max_len = 0
    for i, token in enumerate(spacy_sent):
        if max_len < len(headList_to_root(token)):
            max_len = len(headList_to_root(token))
    return max_len

def lenMax_amongSubtreeList_toRoot(spacy_sent):
    max_len = 0
    for i, token in enumerate(spacy_sent):
        if max_len < len(list(token.subtree)):
            max_len = len(list(token.subtree))
    return max_len


def normalization(max, min, value):
    normalized_value = (value - min) / (max - min) # normalizing the range of value from zero to one.
    return [normalized_value] # list


def FE_dependency(j, sentence, triple_list, outbounder_or_inbounder, dict_dependency):

    feature_dependency = [0] * len(dict_dependency)

    if outbounder_or_inbounder == 0: # inbounder
        for triple in triple_list:
            outbounder = triple[0][0]
            dependency = triple[1]

            if sentence[j] == outbounder:
                feature_dependency[dict_dependency[dependency]] = 1

    else: # inbounder
        for triple in triple_list:
            dependency = triple[1]
            inbounder = triple[2][0]

            if sentence[j] == inbounder:
                feature_dependency[dict_dependency[dependency]] = 1

    return feature_dependency

def normalized_list_n_dependencies(list_n_dependencies):

    min_value = 0 # min(list_n_dependencies)
    max_value = max(list_n_dependencies)

    normalized_list = [0] * len(list_n_dependencies)
    for i, value in enumerate(list_n_dependencies):
        normalized_list[i] = (list_n_dependencies[i] - min_value) / (max_value - min_value)

    return normalized_list

def vectorizing_using_list_dependencies(j_plus_alpha, sentence, normalized_li_n_dep):

    # exception for first
    if j_plus_alpha < 0:
        return [0]

    # exception for last
    try:
        sentence[j_plus_alpha]
    except IndexError:
        return [0]

    # good
    return [normalized_li_n_dep[j_plus_alpha]]



def FE_n_dependencies(j, sentence, normalized_li_n_dep, window_size_n_dep):

    if window_size_n_dep == 1:
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        return t__0

    elif window_size_n_dep == 3:
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        return t_m1 + t__0 + t_p1

    elif window_size_n_dep == 5:
        t_m2 = vectorizing_using_list_dependencies(j-2, sentence, normalized_li_n_dep)
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        t_p2 = vectorizing_using_list_dependencies(j+2, sentence, normalized_li_n_dep)
        return t_m2 + t_m1 + t__0 + t_p1 + t_p2

    elif window_size_n_dep == 7:
        t_m3 = vectorizing_using_list_dependencies(j-3, sentence, normalized_li_n_dep)
        t_m2 = vectorizing_using_list_dependencies(j-2, sentence, normalized_li_n_dep)
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        t_p2 = vectorizing_using_list_dependencies(j+2, sentence, normalized_li_n_dep)
        t_p3 = vectorizing_using_list_dependencies(j+3, sentence, normalized_li_n_dep)
        return t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3

    elif window_size_n_dep == 9:
        t_m4 = vectorizing_using_list_dependencies(j-4, sentence, normalized_li_n_dep)
        t_m3 = vectorizing_using_list_dependencies(j-3, sentence, normalized_li_n_dep)
        t_m2 = vectorizing_using_list_dependencies(j-2, sentence, normalized_li_n_dep)
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        t_p2 = vectorizing_using_list_dependencies(j+2, sentence, normalized_li_n_dep)
        t_p3 = vectorizing_using_list_dependencies(j+3, sentence, normalized_li_n_dep)
        t_p4 = vectorizing_using_list_dependencies(j+4, sentence, normalized_li_n_dep)
        return t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4

    elif window_size_n_dep == 11:
        t_m5 = vectorizing_using_list_dependencies(j-5, sentence, normalized_li_n_dep)
        t_m4 = vectorizing_using_list_dependencies(j-4, sentence, normalized_li_n_dep)
        t_m3 = vectorizing_using_list_dependencies(j-3, sentence, normalized_li_n_dep)
        t_m2 = vectorizing_using_list_dependencies(j-2, sentence, normalized_li_n_dep)
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        t_p2 = vectorizing_using_list_dependencies(j+2, sentence, normalized_li_n_dep)
        t_p3 = vectorizing_using_list_dependencies(j+3, sentence, normalized_li_n_dep)
        t_p4 = vectorizing_using_list_dependencies(j+4, sentence, normalized_li_n_dep)
        t_p5 = vectorizing_using_list_dependencies(j+5, sentence, normalized_li_n_dep)
        return t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5

    elif window_size_n_dep == 13:
        t_m6 = vectorizing_using_list_dependencies(j-6, sentence, normalized_li_n_dep)
        t_m5 = vectorizing_using_list_dependencies(j-5, sentence, normalized_li_n_dep)
        t_m4 = vectorizing_using_list_dependencies(j-4, sentence, normalized_li_n_dep)
        t_m3 = vectorizing_using_list_dependencies(j-3, sentence, normalized_li_n_dep)
        t_m2 = vectorizing_using_list_dependencies(j-2, sentence, normalized_li_n_dep)
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        t_p2 = vectorizing_using_list_dependencies(j+2, sentence, normalized_li_n_dep)
        t_p3 = vectorizing_using_list_dependencies(j+3, sentence, normalized_li_n_dep)
        t_p4 = vectorizing_using_list_dependencies(j+4, sentence, normalized_li_n_dep)
        t_p5 = vectorizing_using_list_dependencies(j+5, sentence, normalized_li_n_dep)
        t_p6 = vectorizing_using_list_dependencies(j+6, sentence, normalized_li_n_dep)
        return t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6

    elif window_size_n_dep == 15:
        t_m7 = vectorizing_using_list_dependencies(j-7, sentence, normalized_li_n_dep)
        t_m6 = vectorizing_using_list_dependencies(j-6, sentence, normalized_li_n_dep)
        t_m5 = vectorizing_using_list_dependencies(j-5, sentence, normalized_li_n_dep)
        t_m4 = vectorizing_using_list_dependencies(j-4, sentence, normalized_li_n_dep)
        t_m3 = vectorizing_using_list_dependencies(j-3, sentence, normalized_li_n_dep)
        t_m2 = vectorizing_using_list_dependencies(j-2, sentence, normalized_li_n_dep)
        t_m1 = vectorizing_using_list_dependencies(j-1, sentence, normalized_li_n_dep)
        t__0 = vectorizing_using_list_dependencies(j, sentence, normalized_li_n_dep)
        t_p1 = vectorizing_using_list_dependencies(j+1, sentence, normalized_li_n_dep)
        t_p2 = vectorizing_using_list_dependencies(j+2, sentence, normalized_li_n_dep)
        t_p3 = vectorizing_using_list_dependencies(j+3, sentence, normalized_li_n_dep)
        t_p4 = vectorizing_using_list_dependencies(j+4, sentence, normalized_li_n_dep)
        t_p5 = vectorizing_using_list_dependencies(j+5, sentence, normalized_li_n_dep)
        t_p6 = vectorizing_using_list_dependencies(j+6, sentence, normalized_li_n_dep)
        t_p7 = vectorizing_using_list_dependencies(j+7, sentence, normalized_li_n_dep)
        return t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7


############################################
""" rule feature extraction """
############################################
def add_rule(FE3_X_train, FE3_X_test, pre_X_train, pre_X_test):
    before_fe = len(FE3_X_train[0][0])
    for i, sentence in enumerate(pre_X_train): # for training data
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)

        for j, token in enumerate(sentence):
            FE3_X_train[i][j] += check_wouldbenice_pattern(detokenized_sent)
            FE3_X_train[i][j] += is_auxVerb_leftside(j, sentence)
            FE3_X_train[i][j] += is_sothatORif_leftside(j, sentence)

            # for sub action, sub obect, ...
            FE3_X_train[i][j] += is_causative_verb(j, sentence) # 사역동사
            FE3_X_train[i][j] += is_abstract_noun(j, sentence) # 추상명사

    for i, sentence in enumerate(pre_X_test): # for training data
        detokenized_sent = detokenizer.detokenize(sentence, return_str=True)

        for j, token in enumerate(sentence):
            FE3_X_test[i][j] += check_wouldbenice_pattern(detokenized_sent)
            FE3_X_test[i][j] += is_auxVerb_leftside(j, sentence)
            FE3_X_test[i][j] += is_sothatORif_leftside(j, sentence)

            # for sub action, sub obect, ...
            FE3_X_test[i][j] += is_causative_verb(j, sentence) # 사역동사
            FE3_X_test[i][j] += is_abstract_noun(j, sentence) # 추상명사
    after_fe = len(FE3_X_train[0][0])
    print('>> rule feature vector dim = ', after_fe-before_fe)
    return FE3_X_train, FE3_X_test

############################################
""" rule feature design """
############################################

def is_causative_verb(j, sentence):
    causative_verb_list = ['allow']
    for token in sentence:
        if token in causative_verb_list:
            return [1]
    return [0]

def is_abstract_noun(j, sentence):
    abstract_noun_list = ['ability']
    for token in sentence:
        if token in abstract_noun_list:
            return [1]
    return [0]

def is_auxVerb_leftside(j_alpha, sentence):

    auxiliary_verb_set = ['shall', 'can', 'want', 'would', 'could', 'should', 'must', 'will', 'may', 'might']

    for j in range(0, j_alpha):
        if sentence[j] in auxiliary_verb_set:
            return [1]
            break
    return [0]

def check_wouldbenice_pattern(detokenized_sentence):

    if detokenized_sentence.find('be able to') != -1:
        return [1]
    elif detokenized_sentence.find('it would be') != -1:
        return [1]
    elif detokenized_sentence.find('would be nice') != -1:
        return [1]
    elif detokenized_sentence.find('it is possible') != -1:
        return [1]
    elif detokenized_sentence.find('want') != -1:
        return [1]
    elif detokenized_sentence.find('will be nice') != -1:
        return [1]
    elif detokenized_sentence.find('could be nice') != -1:
        return [1]
    elif detokenized_sentence.find('will really nice') != -1:
        return [1]
    elif detokenized_sentence.find('would like') != -1:
        return [1]
    else:
        return [0]

def is_auxVerb_leftside(j_alpha, sentence):

    auxiliary_verb_set = ['shall', 'can', 'want', 'would', 'could', 'should', 'must', 'will', 'may', 'might']

    for j in range(0, j_alpha):
        if sentence[j] in auxiliary_verb_set:
            return [1]
            break
    return [0]


def is_sothatORif_leftside(j_alpha, sentence):

    featureVec = [0, 0]
    sothatCheck = False
    sothatIndex = -1
    ifCheck = False
    ifIndex = -1

    for j in range(0, j_alpha):
        if not j == len(sentence)-1:
            if sentence[j] == 'so' and sentence[j+1] == 'that':
                sothatCheck = True
                sothatIndex = j

    for j in range(0, j_alpha):
        if sentence[j] == 'if' or sentence[j] == 'when': # when
            ifCheck = True
            ifIndex = j

    if sothatCheck == True and ifCheck == True:
        if sothatIndex > ifIndex:
            return [1, 0]
        else:
            return [0, 1]
    elif sothatCheck == True and ifCheck == False:
        return [1, 0]
    elif sothatCheck == False and ifCheck == True:
        return [0, 1]
    else:
        return [0, 0]

def convert_dummy(value):
    dummy_feature = [0] * 10 # 10 stage
    if value <1:
        dummy_feature[0] = 1
    elif value >=1 and value <2:
        dummy_feature[1] = 1
    elif value >=2 and value <3:
        dummy_feature[2] = 1
    elif value >=3 and value <4:
        dummy_feature[3] = 1
    elif value >=4 and value <5:
        dummy_feature[4] = 1
    elif value >=5 and value <6:
        dummy_feature[5] = 1
    elif value >=6 and value <7:
        dummy_feature[6] = 1
    elif value >=7 and value <8:
        dummy_feature[7] = 1
    elif value >=8 and value <9:
        dummy_feature[8] = 1
    elif value >=9 and value <10:
        dummy_feature[9] = 1
    return dummy_feature

def position_which(sentence):

    # which가 문장에서 여러번 나올 수 있으나
    # 그냥 앞에서부터 추적해서 제일 먼저 나온 which의 position을 return한다.
    for i, token in enumerate(sentence):
        if token == 'which':
            return i
        else:
            return -1 # which가 없다.

def FE_which_left_right(j, index_which):
    if index_which == -1:
        return [0, 0]
    elif j < index_which:
        return [1, 0]
    else:
        return [0, 1]


def position_sothat(sentence):

    for i, token in enumerate(sentence):
        if sentence[i] == 'so' and sentence[i+1] == 'that':
            return i
        else:
            return -1 # so that이 없다.

def FE_sothat_left_right(j, index_sothat):
    if index_sothat == -1:
        return [0, 0]
    elif j < index_sothat:
        return [1, 0]
    else:
        return [0, 1]

def vectorized_token_using_prep_lookup(j, sentence):

    list_prep = [
         'in','at','on','to','of','for','from','by'
    ]
    lookup_prep = { }
    for i, prep in enumerate(list_prep):
        lookup_prep[prep] = i


    # token feature 0으로 초기화
    prep_feature = [0] * len(lookup_prep)

    # 자신이 전치사인지
    if sentence[j] in list_prep:
        prep_feature[lookup_prep[sentence[j]]] = 1
        return prep_feature

    # 자신이 전치사가 아니라면..
    # 왼쪽으로 0 index까지 전치사를 찾는다. 없으면 0
    else:
        for k in range(j, -1, -1): # j에서 0까지 거꾸로 for loop를 돌린다.
            if sentence[k] in list_prep:
                prep_feature[lookup_prep[sentence[k]]] = 1
                return prep_feature

    # 그래도 전치사가 없다면 그냥 0벡터 return
    return prep_feature

############################################
""" stanford feature extraction """
############################################
def create_lookup_pos():
    list_POS = [
        'DT','JJ','NN','MD','VB',',','CC','IN','.','VBG','VBZ','VBN','TO','WRB','NNS','RB','PRP','VBP','PRP$','CD',
        'NNP','EX','VBD','JJR',':','WDT','RP','NNPS','POS',"''",'RBR','WP','PDT','JJS','RBS','$','FW'
    ]
    lookup_table_POS = { }
    for i, token in enumerate(list_POS):
        lookup_table_POS[token] = i # POS에서는 1부터(i+1) 시작안해도 된다. POS에는 unknown이 없기 때문이다.
    # Add unknown token
    print('>> stanford pos lookup table dim = ', len(lookup_table_POS))
    return lookup_table_POS

def add_pos(FE2_X_train, FE2_X_test, Xtrain, Xtest, lookup_pos, window_size_pos):
    before_fe = len(FE2_X_train[0][0])
    # For training data
    for i, sentence in enumerate(Xtrain):
        pos_tagged_sentence = nltk.pos_tag(sentence)
        #print(pos_tagged_sentence)
        for j, tuple_ in enumerate(pos_tagged_sentence): # pos_sentence
            FE2_X_train[i][j] += FE_POS(j, pos_tagged_sentence, window_size_pos, lookup_pos)

    # For testing data
    for i, sentence in enumerate(Xtest):
        pos_tagged_sentence = nltk.pos_tag(sentence)
        for j, tuple_ in enumerate(pos_tagged_sentence): # pos_sentence
            FE2_X_test[i][j] += FE_POS(j, pos_tagged_sentence, window_size_pos, lookup_pos)
    after_fe = len(FE2_X_train[0][0])
    print('>> pos feature vector dim = ', after_fe-before_fe)
    return FE2_X_train, FE2_X_test

def add_constituency(FE2_X_train, FE2_X_test, Xtrain, Xtest, window_size_depth, window_size_n_siblings):
    global temp_sentence

    before_fe = len(FE2_X_train[0][0])
    ### for training set
    cnt = 0
    temp_depth_list = []
    temp_n_sibling_list = []
    for i, sentence in enumerate(Xtrain):
        temp = nltk.pos_tag(sentence)
        par_result = parser.tagged_parse(temp)
        for tree in par_result:
            parse_tree = tree
        #print(parse_tree)
        # For PP, NP, VP,...
        temp_sentence = sentence
        initial_tree_parameter(sentence)
        traverseTree(parse_tree) # For getting tree information

        # For depth, n_sibling...
        depth_list, n_siblings_list = tree_depth_and_n_sibling(parse_tree)
        assert(len(sentence) == len(depth_list))
    #     print(sentence)
    #     print(parse_tree.leaves())
        normalized_li_depth = normalizing_depth_list(depth_list)
        normalized_li_n_siblings = normalizing_n_siblings_list(n_siblings_list)

        for j, token in enumerate(sentence):
            ### Tree depth and sibling
            FE2_X_train[i][j] += FE_depth(j, sentence, normalized_li_depth, window_size_depth)
            FE2_X_train[i][j] += FE_n_siblings(j, sentence, normalized_li_n_siblings, window_size_n_siblings)
            ### Phrases-based Features
#             FE2_X_train[i][j] += [list_NPSBAR[j]] # defalut 오직 SBAR만 뽑는다.
#             FE2_X_train[i][j] += [list_VPSBAR[j]]
#             FE2_X_train[i][j] += [list_SSBAR[j]]
#             FE2_X_train[i][j] += [list_SQSBAR[j]]

#             FE2_X_train[i][j] += [list_PPPP[j]] # default 오직 PP만 뽑는다.
#             FE2_X_train[i][j] += [list_VPPP[j]]
#             FE2_X_train[i][j] += [list_NPPP[j]] # sub_refinement_of_object를 추출하는데 도움..

#             FE2_X_train[i][j] += [list_NPNP[j]]
#             FE2_X_train[i][j] += [list_VPNP[j]]
#             FE2_X_train[i][j] += [list_SNP[j]]

#             FE2_X_train[i][j] += [list_VPVP[j]]
#             FE2_X_train[i][j] += [list_SVP[j]]
    #     cnt += 1
    #     print('===================================================================')
    #     if cnt == 1:
    #         break

    ### for test set
    cnt = 0
    for i, sentence in enumerate(Xtest):
        temp = nltk.pos_tag(sentence)
        par_result = parser.tagged_parse(temp)
        for tree in par_result:
            parse_tree = tree
        #print(parse_tree)

        # For PP, NP, VP, ...
        temp_sentence = sentence
        initial_tree_parameter(sentence)
        traverseTree(parse_tree) # For getting tree information

        # For dpeth, n_sibling
        depth_list, n_siblings_list = tree_depth_and_n_sibling(parse_tree)
        assert(len(sentence) == len(depth_list))
    #     print(sentence)
    #     print(parse_tree.leaves())
        normalized_li_depth = normalizing_depth_list(depth_list)
        normalized_li_n_siblings = normalizing_n_siblings_list(n_siblings_list)

        for j, token in enumerate(sentence):
            ### Tree-depth, Sibling
            FE2_X_test[i][j] += FE_depth(j, sentence, normalized_li_depth, window_size_depth)
            FE2_X_test[i][j] += FE_n_siblings(j, sentence, normalized_li_n_siblings, window_size_n_siblings)
            ## Phrases-based Festures
#             FE2_X_test[i][j] += [list_NPSBAR[j]]
#             FE2_X_test[i][j] += [list_VPSBAR[j]]
#             FE2_X_test[i][j] += [list_SSBAR[j]]
#             FE2_X_test[i][j] += [list_SQSBAR[j]]

#             FE2_X_test[i][j] += [list_PPPP[j]]
#             FE2_X_test[i][j] += [list_VPPP[j]]
#             FE2_X_test[i][j] += [list_NPPP[j]]

#             FE2_X_test[i][j] += [list_NPNP[j]]
#             FE2_X_test[i][j] += [list_VPNP[j]]
#             FE2_X_test[i][j] += [list_SNP[j]]

#             FE2_X_test[i][j] += [list_VPVP[j]]
#             FE2_X_test[i][j] += [list_SVP[j]]

    #     print(convert_dummy2(depth_list[j], 0))
    #     print(convert_dummy2(n_siblings_list[j], 1))
    #     print([list_SBAR[j]])
    #     print([list_VPPP[j]])
    #     print([list_NPPP[j]])

    #     cnt += 1
    #     print('===================================================================')
    #     if cnt == 1:
    #         break
    after_fe = len(FE2_X_train[0][0])
    print('>> constituency parser feature vector dim = ', after_fe-before_fe)
    return FE2_X_train, FE2_X_test


############################################
""" stanford feature design """
############################################

### Features for several phrases
def initial_tree_parameter(temp_sentence):

    global list_current_depth
    global previous_height
    global current_height
    global previous_depth
    global current_depth
    global phrase_list_until_leaf
    global per_token
    global depth_cnt

    ######
    global index_NPSBAR
    global index_VPSBAR
    global index_SSBAR
    global index_SQSBAR

    global index_PPPP
    global index_VPPP
    global index_NPPP

    global index_NPNP
    global index_VPNP
    global index_SNP

    global index_VPVP
    global index_SVP

    global list_NPSBAR
    global list_VPSBAR
    global list_SSBAR
    global list_SQSBAR

    global list_PPPP
    global list_VPPP
    global list_NPPP

    global list_NPNP
    global list_VPNP
    global list_SNP

    global list_VPVP
    global list_SVP

    list_NPSBAR = [0] * len(temp_sentence)
    list_VPSBAR = [0] * len(temp_sentence)
    list_SSBAR = [0] * len(temp_sentence)
    list_SQSBAR = [0] * len(temp_sentence)

    list_PPPP = [0] * len(temp_sentence)
    list_VPPP = [0] * len(temp_sentence)
    list_NPPP = [0] * len(temp_sentence)

    list_NPNP = [0] * len(temp_sentence)
    list_VPNP = [0] * len(temp_sentence)
    list_SNP = [0] * len(temp_sentence)

    list_VPVP = [0] * len(temp_sentence)
    list_SVP = [0] * len(temp_sentence)

    phrase_list_until_leaf = []
    per_depth = []
    per_token = []
    list_current_depth = []
    previous_height = 0
    current_height = 0
    previous_depth = 0
    current_depth = 0
    depth_cnt = 0

    index_NPSBAR = -1
    index_VPSBAR = -1
    index_SSBAR = -1
    index_SQSBAR = -1

    index_PPPP = -1
    index_VPPP = -1
    index_NPPP = -1

    index_NPNP = -1
    index_VPNP = -1
    index_SNP = -1

    index_VPVP = -1
    index_SVP = -1


def traverseTree(tree):

    global list_current_depth
    global previous_height
    global current_height
    global previous_depth
    global current_depth
    global phrase_list_until_leaf
    global per_token
    global depth_cnt
    global temp_sentence

    #####
    global index_NPSBAR
    global index_VPSBAR
    global index_SSBAR
    global index_SQSBAR

    global index_PPPP
    global index_VPPP
    global index_NPPP

    global index_NPNP
    global index_VPNP
    global index_SNP

    global index_VPVP
    global index_SVP

    ######
    global list_NPSBAR
    global list_VPSBAR
    global list_SSBAR
    global list_SQSBAR

    global list_PPPP
    global list_VPPP
    global list_NPPP

    global list_NPNP
    global list_VPNP
    global list_SNP

    global list_VPVP
    global list_SVP


    if type(tree) == nltk.tree.Tree:

        # Initialication
        current_height = tree.height()


        # For SBAR, NPPP, VPPP, NP
#         if tree.label() == 'SBAR':
#             tokens_SBAR = tree.leaves()
# #             print(len(tokens_SBAR))
# #             print(len(temp_sentence))
# #             print(tokens_SBAR)
# #             print(temp_sentence)
# #             print("**************************")
#             for i, token in enumerate(temp_sentence):
#                 if temp_sentence[i] == tokens_SBAR[0]:
#                     if temp_sentence[i+len(tokens_SBAR)-1] == tokens_SBAR[-1]:
#                         index_SBAR = i
#                         break

#             if not index_SBAR == -1:
#                 for j in range(index_SBAR, index_SBAR + len(tokens_SBAR)):
#                     list_SBAR[j] = 1

        # NP - SBAR
        if tree.label() == 'SBAR' and list_current_depth[-1] == 'NP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_NPSBAR = i
                    break
            if not index_NPSBAR == -1:
                for j in range(index_NPSBAR, index_NPSBAR+len(tree.leaves())):
                    list_NPSBAR[j] = 1

        # VP - SBAR
        elif tree.label() == 'SBAR' and list_current_depth[-1] == 'VP':

            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_VPSBAR = i
                    break
            if not index_VPSBAR == -1:
                for j in range(index_VPSBAR, index_VPSBAR+len(tree.leaves())):
                    list_VPSBAR[j] = 1

        # S - SBAR
        elif tree.label() == 'SBAR' and list_current_depth[-1] == 'S':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_SSBAR = i
                    break
            if not index_SSBAR == -1:
                for j in range(index_SSBAR, index_SSBAR+len(tree.leaves())):
                    list_SSBAR[j] = 1

        # SQ - SBAR
        elif tree.label() == 'SBAR' and list_current_depth[-1] == 'SQ':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_SQSBAR = i
                    break
            if not index_SQSBAR == -1:
                for j in range(index_SQSBAR, index_SQSBAR+len(tree.leaves())):
                    list_SQSBAR[j] = 1


        # PP - PP
        elif tree.label() == 'PP' and list_current_depth[-1] == 'PP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_PPPP = i
                    break
            if not index_PPPP == -1:
                for j in range(index_PPPP, index_PPPP+len(tree.leaves())):
                    list_PPPP[j] = 1

        # VP - PP
        elif tree.label() == 'PP' and list_current_depth[-1] == 'VP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_VPPP = i
                    break
            if not index_VPPP == -1:
                for j in range(index_VPPP, index_VPPP+len(tree.leaves())):
                    list_VPPP[j] = 1

        # NP - PP
        elif tree.label() == 'PP' and list_current_depth[-1] == 'NP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_NPPP = i
                    break
            if not index_NPPP == -1:
                for j in range(index_NPPP, index_NPPP+len(tree.leaves())):
                    list_NPPP[j] = 1

        # NP - NP
        elif tree.label() == 'NP' and list_current_depth[-1] == 'NP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_NPNP = i
                    break
            if not index_NPNP == -1:
                for j in range(index_NPNP, index_NPNP+len(tree.leaves())):
                    list_NPNP[j] = 1

        # VP - NP
        elif tree.label() == 'NP' and list_current_depth[-1] == 'VP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_VPNP = i
                    break
            if not index_VPNP == -1:
                for j in range(index_VPNP, index_VPNP+len(tree.leaves())):
                    list_VPNP[j] = 1

        # S - NP
        elif tree.label() == 'NP' and list_current_depth[-1] == 'S':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_SNP = i
                    break
            if not index_SNP == -1:
                for j in range(index_SNP, index_SNP+len(tree.leaves())):
                    list_SNP[j] = 1

        # VP - VP
        elif tree.label() == 'VP' and list_current_depth[-1] == 'VP':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_VPVP = i
                    break
            if not index_VPVP == -1:
                for j in range(index_VPVP, index_VPVP+len(tree.leaves())):
                    list_VPVP[j] = 1

        # S - VP
        elif tree.label() == 'VP' and list_current_depth[-1] == 'S':
            for i, token in enumerate(temp_sentence):
                if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    index_SVP = i
                    break
            if not index_SVP == -1:
                for j in range(index_SVP, index_SVP+len(tree.leaves())):
                    list_SVP[j] = 1

#         elif tree.label() == 'NP':
#             tokens_NP = tree.leaves()

#             for i, token in enumerate(temp_sentence):
#                 if temp_sentence[i] == tokens_NP[0] and temp_sentence[i+len(tokens_NP)-1] == tokens_NP[-1]:
#                     index_NP = i

#             if not index_NP == -1:
#                 for j in range(index_NP, index_NP + len(tokens_NP)):
#                     list_NP[j] = 1

        ### For Depth
        # fist tree search
        if depth_cnt == 0:
            current_depth = current_height # depth initialization
            list_current_depth.append(tree.label())
            depth_cnt += 1
        # what is the current depth?
        if current_height == previous_depth-1:
            current_depth = current_height
            list_current_depth.append(tree.label())

#         if current_height > previous_height:
#             per_token = []
#         if current_height == 2:
#             phrase_list_until_leaf.append(per_token)
#         else:
#             per_token.append(tree.label())

        previous_height = current_height
        previous_depth = current_depth


#         print("tree:", tree)
#         print("======> tree.pos()", tree.pos())
#         print("======> tree.height()", tree.height())
#         print("======> tree.label()", tree.label())
#         print("======> tree.leaves()", tree.leaves())
#         print("======> current_depth", current_depth)
#         print('======> current depth list', list_current_depth)

#         print('*************************************************************************************************')
#         print('\n')

    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverseTree(subtree) # recursive


##### pos
def vectorized_token_using_pos_lookup(j_plus_alpha, sentence, lookup):
    # token feature 0으로 초기화
    token_feature = [0] * len(lookup)
    ### 1. for array index exception
    if j_plus_alpha < 0: # for first index exception
        return token_feature

    try:
        sentence[j_plus_alpha] # f.;/or last index exception
    except IndexError:
        return token_feature # 모든 element가 0인 token feature

    ### 2. for key (unknown token) exception
    # in pos, there is no exception for lookup table
#     try:
#         lookup[sentence[j_plus_alpha]]
#     except KeyError:
#         if unknown == True:
#             token_feature[lookup['unknown_token']-1] = 1
#             return token_feature # which is unknown index
#         else: # unknown token 적용안할 것이다...
#             return token_feature # 따라서, 그냥 0벡터로..

    ### 3. oridinal case
#     token_feature[lookup[sentence[j_plus_alpha]]-1] = 1 # unknown pos가 없으니 -1은 하지 않는다.
    token_feature[lookup[sentence[j_plus_alpha][1]]] = 1
    return token_feature

### Features for tree depth and number of siblings
def tree_depth_and_n_sibling(tree):

    n_leaves = len(tree.leaves())
    leave_pos = list(tree.leaf_treeposition(n) for n in range(n_leaves))

    # list for depth
    depth_list = [0] * len(leave_pos)
    for i, each_pos in enumerate(leave_pos):
        depth_list[i] = len(each_pos)

    # list for number of siblings
    n_siblings_list = [0] * len(leave_pos)

    sibling_freq_dict = dict(collections.Counter(depth_list)) # if same depth -> it is sibling.

    for i, depth in enumerate(depth_list):
        n_siblings_list[i] = sibling_freq_dict[depth]

    return depth_list, n_siblings_list


def convert_normalization(count, depth_or_sibling):

    if depth_or_sibling == 0: # depth
        assert(count <= 35 and count >= 3)
        min_depth = 3
        max_depth = 35
        return [(count - min_depth) / (max_depth - min_depth)] # list return


    else: # sibling
        assert(count <= 16 and count >= 1)
        min_depth = 1
        max_depth = 16
        return [(count - min_depth) / (max_depth - min_depth)] # list return

def normalizing_depth_list(depth_list):

    min_value = 2
    max_value = 35

    normalized_list = [0] * len(depth_list)
    for i, value in enumerate(depth_list):
        normalized_list[i] = (depth_list[i] - min_value) / (max_value - min_value)

    return normalized_list

def vectorizing_using_list_depth(j_plus_alpha, sentence, normalized_li_depth):

    # exception for first
    if j_plus_alpha < 0:
        return [0]

    # exception for last
    try:
        sentence[j_plus_alpha]
    except IndexError:
        return [0]

    # all pass...
    return [normalized_li_depth[j_plus_alpha]]

def FE_depth(j, sentence, normalized_li_depth, window_size_depth):

    if window_size_depth == 1:
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        return t__0

    elif window_size_depth == 3:
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        return t_m1 + t__0 + t_p1

    elif window_size_depth == 5:
        t_m2 = vectorizing_using_list_depth(j-2, sentence, normalized_li_depth)
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        t_p2 = vectorizing_using_list_depth(j+2, sentence, normalized_li_depth)
        return t_m2 + t_m1 + t__0 + t_p1 + t_p2

    elif window_size_depth == 7:
        t_m3 = vectorizing_using_list_depth(j-3, sentence, normalized_li_depth)
        t_m2 = vectorizing_using_list_depth(j-2, sentence, normalized_li_depth)
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        t_p2 = vectorizing_using_list_depth(j+2, sentence, normalized_li_depth)
        t_p3 = vectorizing_using_list_depth(j+3, sentence, normalized_li_depth)
        return t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3

    elif window_size_depth == 9:
        t_m4 = vectorizing_using_list_depth(j-4, sentence, normalized_li_depth)
        t_m3 = vectorizing_using_list_depth(j-3, sentence, normalized_li_depth)
        t_m2 = vectorizing_using_list_depth(j-2, sentence, normalized_li_depth)
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        t_p2 = vectorizing_using_list_depth(j+2, sentence, normalized_li_depth)
        t_p3 = vectorizing_using_list_depth(j+3, sentence, normalized_li_depth)
        t_p4 = vectorizing_using_list_depth(j+4, sentence, normalized_li_depth)
        return t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4

    elif window_size_depth == 11:
        t_m5 = vectorizing_using_list_depth(j-5, sentence, normalized_li_depth)
        t_m4 = vectorizing_using_list_depth(j-4, sentence, normalized_li_depth)
        t_m3 = vectorizing_using_list_depth(j-3, sentence, normalized_li_depth)
        t_m2 = vectorizing_using_list_depth(j-2, sentence, normalized_li_depth)
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        t_p2 = vectorizing_using_list_depth(j+2, sentence, normalized_li_depth)
        t_p3 = vectorizing_using_list_depth(j+3, sentence, normalized_li_depth)
        t_p4 = vectorizing_using_list_depth(j+4, sentence, normalized_li_depth)
        t_p5 = vectorizing_using_list_depth(j+5, sentence, normalized_li_depth)
        return t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5

    elif window_size_depth == 13:
        t_m6 = vectorizing_using_list_depth(j-6, sentence, normalized_li_depth)
        t_m5 = vectorizing_using_list_depth(j-5, sentence, normalized_li_depth)
        t_m4 = vectorizing_using_list_depth(j-4, sentence, normalized_li_depth)
        t_m3 = vectorizing_using_list_depth(j-3, sentence, normalized_li_depth)
        t_m2 = vectorizing_using_list_depth(j-2, sentence, normalized_li_depth)
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        t_p2 = vectorizing_using_list_depth(j+2, sentence, normalized_li_depth)
        t_p3 = vectorizing_using_list_depth(j+3, sentence, normalized_li_depth)
        t_p4 = vectorizing_using_list_depth(j+4, sentence, normalized_li_depth)
        t_p5 = vectorizing_using_list_depth(j+5, sentence, normalized_li_depth)
        t_p6 = vectorizing_using_list_depth(j+6, sentence, normalized_li_depth)
        return t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6

    elif window_size_depth == 15:
        t_m7 = vectorizing_using_list_depth(j-7, sentence, normalized_li_depth)
        t_m6 = vectorizing_using_list_depth(j-6, sentence, normalized_li_depth)
        t_m5 = vectorizing_using_list_depth(j-5, sentence, normalized_li_depth)
        t_m4 = vectorizing_using_list_depth(j-4, sentence, normalized_li_depth)
        t_m3 = vectorizing_using_list_depth(j-3, sentence, normalized_li_depth)
        t_m2 = vectorizing_using_list_depth(j-2, sentence, normalized_li_depth)
        t_m1 = vectorizing_using_list_depth(j-1, sentence, normalized_li_depth)
        t__0 = vectorizing_using_list_depth(j, sentence, normalized_li_depth)
        t_p1 = vectorizing_using_list_depth(j+1, sentence, normalized_li_depth)
        t_p2 = vectorizing_using_list_depth(j+2, sentence, normalized_li_depth)
        t_p3 = vectorizing_using_list_depth(j+3, sentence, normalized_li_depth)
        t_p4 = vectorizing_using_list_depth(j+4, sentence, normalized_li_depth)
        t_p5 = vectorizing_using_list_depth(j+5, sentence, normalized_li_depth)
        t_p6 = vectorizing_using_list_depth(j+6, sentence, normalized_li_depth)
        t_p7 = vectorizing_using_list_depth(j+7, sentence, normalized_li_depth)
        return t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7

def normalizing_n_siblings_list(n_siblings_list):

    min_value = 0
    max_value = 16

    normalized_list = [0] * len(n_siblings_list)
    for i, value in enumerate(n_siblings_list):
        normalized_list[i] = (n_siblings_list[i] - min_value) / (max_value - min_value)

    return normalized_list


def vectorizing_using_list_n_siblings(j_plus_alpha, sentence, normalized_li_n_siblings):

    # exception for first
    if j_plus_alpha < 0:
        return [0]

    # exception for last
    try:
        sentence[j_plus_alpha]
    except IndexError:
        return [0]

    # all pass...
    return [normalized_li_n_siblings[j_plus_alpha]]


def FE_n_siblings(j, sentence, normalized_li_n_siblings, window_size_n_siblings):

    if window_size_n_siblings == 1:
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        return t__0

    elif window_size_n_siblings == 3:
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        return t_m1 + t__0 + t_p1

    elif window_size_n_siblings == 5:
        t_m2 = vectorizing_using_list_n_siblings(j-2, sentence, normalized_li_n_siblings)
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        t_p2 = vectorizing_using_list_n_siblings(j+2, sentence, normalized_li_n_siblings)
        return t_m2 + t_m1 + t__0 + t_p1 + t_p2

    elif window_size_n_siblings == 7:
        t_m3 = vectorizing_using_list_n_siblings(j-3, sentence, normalized_li_n_siblings)
        t_m2 = vectorizing_using_list_n_siblings(j-2, sentence, normalized_li_n_siblings)
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        t_p2 = vectorizing_using_list_n_siblings(j+2, sentence, normalized_li_n_siblings)
        t_p3 = vectorizing_using_list_n_siblings(j+3, sentence, normalized_li_n_siblings)
        return t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3

    elif window_size_n_siblings == 9:
        t_m4 = vectorizing_using_list_n_siblings(j-4, sentence, normalized_li_n_siblings)
        t_m3 = vectorizing_using_list_n_siblings(j-3, sentence, normalized_li_n_siblings)
        t_m2 = vectorizing_using_list_n_siblings(j-2, sentence, normalized_li_n_siblings)
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        t_p2 = vectorizing_using_list_n_siblings(j+2, sentence, normalized_li_n_siblings)
        t_p3 = vectorizing_using_list_n_siblings(j+3, sentence, normalized_li_n_siblings)
        t_p4 = vectorizing_using_list_n_siblings(j+4, sentence, normalized_li_n_siblings)
        return t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4

    elif window_size_n_siblings == 11:
        t_m5 = vectorizing_using_list_n_siblings(j-5, sentence, normalized_li_n_siblings)
        t_m4 = vectorizing_using_list_n_siblings(j-4, sentence, normalized_li_n_siblings)
        t_m3 = vectorizing_using_list_n_siblings(j-3, sentence, normalized_li_n_siblings)
        t_m2 = vectorizing_using_list_n_siblings(j-2, sentence, normalized_li_n_siblings)
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        t_p2 = vectorizing_using_list_n_siblings(j+2, sentence, normalized_li_n_siblings)
        t_p3 = vectorizing_using_list_n_siblings(j+3, sentence, normalized_li_n_siblings)
        t_p4 = vectorizing_using_list_n_siblings(j+4, sentence, normalized_li_n_siblings)
        t_p5 = vectorizing_using_list_n_siblings(j+5, sentence, normalized_li_n_siblings)
        return t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5

    elif window_size_n_siblings == 13:
        t_m6 = vectorizing_using_list_n_siblings(j-6, sentence, normalized_li_n_siblings)
        t_m5 = vectorizing_using_list_n_siblings(j-5, sentence, normalized_li_n_siblings)
        t_m4 = vectorizing_using_list_n_siblings(j-4, sentence, normalized_li_n_siblings)
        t_m3 = vectorizing_using_list_n_siblings(j-3, sentence, normalized_li_n_siblings)
        t_m2 = vectorizing_using_list_n_siblings(j-2, sentence, normalized_li_n_siblings)
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        t_p2 = vectorizing_using_list_n_siblings(j+2, sentence, normalized_li_n_siblings)
        t_p3 = vectorizing_using_list_n_siblings(j+3, sentence, normalized_li_n_siblings)
        t_p4 = vectorizing_using_list_n_siblings(j+4, sentence, normalized_li_n_siblings)
        t_p5 = vectorizing_using_list_n_siblings(j+5, sentence, normalized_li_n_siblings)
        t_p6 = vectorizing_using_list_n_siblings(j+6, sentence, normalized_li_n_siblings)
        return t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6

    elif window_size_n_siblings == 15:
        t_m7 = vectorizing_using_list_n_siblings(j-7, sentence, normalized_li_n_siblings)
        t_m6 = vectorizing_using_list_n_siblings(j-6, sentence, normalized_li_n_siblings)
        t_m5 = vectorizing_using_list_n_siblings(j-5, sentence, normalized_li_n_siblings)
        t_m4 = vectorizing_using_list_n_siblings(j-4, sentence, normalized_li_n_siblings)
        t_m3 = vectorizing_using_list_n_siblings(j-3, sentence, normalized_li_n_siblings)
        t_m2 = vectorizing_using_list_n_siblings(j-2, sentence, normalized_li_n_siblings)
        t_m1 = vectorizing_using_list_n_siblings(j-1, sentence, normalized_li_n_siblings)
        t__0 = vectorizing_using_list_n_siblings(j, sentence, normalized_li_n_siblings)
        t_p1 = vectorizing_using_list_n_siblings(j+1, sentence, normalized_li_n_siblings)
        t_p2 = vectorizing_using_list_n_siblings(j+2, sentence, normalized_li_n_siblings)
        t_p3 = vectorizing_using_list_n_siblings(j+3, sentence, normalized_li_n_siblings)
        t_p4 = vectorizing_using_list_n_siblings(j+4, sentence, normalized_li_n_siblings)
        t_p5 = vectorizing_using_list_n_siblings(j+5, sentence, normalized_li_n_siblings)
        t_p6 = vectorizing_using_list_n_siblings(j+6, sentence, normalized_li_n_siblings)
        t_p7 = vectorizing_using_list_n_siblings(j+7, sentence, normalized_li_n_siblings)
        return t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7


def convert_dummy(count, depth_or_sibling):

    depth_feature = [0] * 5 # 5 stage
    sibling_feature = [0] * 3 # 3 stage

    if depth_or_sibling == 0: # depth
        if count <=4:
            depth_feature[0] = 1
        elif count >=5 and count <=7:
            depth_feature[1] = 1
        elif count >=8 and count <=12:
            depth_feature[2] = 1
        elif count >=13 and count <=22:
            depth_feature[3] = 1
        else:
            depth_feature[4] = 1
        return depth_feature

    else: # sibling
        if count <=4:
            sibling_feature[0] = 1
        elif count >=5 and count <=9:
            sibling_feature[1] = 1
        else:
            sibling_feature[2] = 1
        return sibling_feature

def convert_dummy2(count, depth_or_sibling):

    depth_feature = [0] * 10 # 10 stage
    sibling_feature = [0] * 10 # 10 stage

    if depth_or_sibling == 0: # depth
        if count <=3:
            depth_feature[0] = 1
        elif count ==4:
            depth_feature[1] = 1
        elif count ==5:
            depth_feature[2] = 1
        elif count ==6:
            depth_feature[3] = 1
        elif count ==7:
            depth_feature[4] = 1
        elif count ==8:
            depth_feature[5] = 1
        elif count >=9 and count <=11:
            depth_feature[6] = 1
        elif count >=12 and count <=15:
            depth_feature[7] = 1
        elif count >=16 and count <=22:
            depth_feature[8] = 1
        elif count >=23:
            depth_feature[9] = 1
        return depth_feature

    else: # sibling
        if count ==1:
            sibling_feature[0] = 1
        elif count ==2:
            sibling_feature[1] = 1
        elif count ==3:
            sibling_feature[2] = 1
        elif count ==4:
            sibling_feature[3] = 1
        elif count ==5:
            sibling_feature[4] = 1
        elif count ==6:
            sibling_feature[5] = 1
        elif count ==7:
            sibling_feature[6] = 1
        elif count ==8:
            sibling_feature[7] = 1
        elif count ==9:
            sibling_feature[8] = 1
        else:
            sibling_feature[9] = 1
        return sibling_feature

def convert_dummy3(count, depth_or_sibling):

    depth_feature = [0] * 32 # 10 stage
    sibling_feature = [0] * 16 # 10 stage

    if depth_or_sibling == 0: # depth
        if count <=3:
            depth_feature[0] = 1
        elif count ==4:
            depth_feature[1] = 1
        elif count ==5:
            depth_feature[2] = 1
        elif count ==6:
            depth_feature[3] = 1
        elif count ==7:
            depth_feature[4] = 1
        elif count ==8:
            depth_feature[6] = 1
        elif count ==9:
            depth_feature[7] = 1
        elif count ==10:
            depth_feature[8] = 1
        elif count ==11:
            depth_feature[9] = 1
        elif count ==12:
            depth_feature[10] = 1
        elif count ==13:
            depth_feature[11] = 1
        elif count ==14:
            depth_feature[12] = 1
        elif count ==15:
            depth_feature[13] = 1
        elif count ==16:
            depth_feature[14] = 1
        elif count ==17:
            depth_feature[15] = 1
        elif count ==18:
            depth_feature[16] = 1
        elif count ==19:
            depth_feature[17] = 1
        elif count ==20:
            depth_feature[18] = 1
        elif count ==21:
            depth_feature[19] = 1
        elif count ==22:
            depth_feature[20] = 1
        elif count ==23:
            depth_feature[21] = 1
        elif count ==24:
            depth_feature[22] = 1
        elif count ==25:
            depth_feature[23] = 1
        elif count ==26:
            depth_feature[24] = 1
        elif count ==27:
            depth_feature[25] = 1
        elif count ==28:
            depth_feature[26] = 1
        elif count ==29:
            depth_feature[27] = 1
        elif count ==30:
            depth_feature[28] = 1
        elif count ==31:
            depth_feature[29] = 1
        elif count ==32:
            depth_feature[30] = 1
        else:
            depth_feature[31] = 1
        return depth_feature

    else: # sibling
        if count ==1:
            sibling_feature[0] = 1
        elif count ==2:
            sibling_feature[1] = 1
        elif count ==3:
            sibling_feature[2] = 1
        elif count ==4:
            sibling_feature[3] = 1
        elif count ==5:
            sibling_feature[4] = 1
        elif count ==6:
            sibling_feature[5] = 1
        elif count ==7:
            sibling_feature[6] = 1
        elif count ==8:
            sibling_feature[7] = 1
        elif count ==9:
            sibling_feature[8] = 1
        elif count ==10:
            sibling_feature[9] = 1
        elif count ==11:
            sibling_feature[10] = 1
        elif count ==12:
            sibling_feature[11] = 1
        elif count ==13:
            sibling_feature[12] = 1
        elif count ==14:
            sibling_feature[13] = 1
        elif count ==15:
            sibling_feature[14] = 1
        else:
            sibling_feature[15] = 1
        return sibling_feature

def FE_POS(j, pos_tagged_sentence, window_size_pos, lookup_table_POS): # Feature extraction using POS

    if window_size_pos == 1:
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        return t__0
    elif window_size_pos == 3:
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        return t_m1 + t__0 + t_p1
    elif window_size_pos == 5:
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        return t_m2 + t_m1 + t__0 + t_p1 + t_p2
    elif window_size_pos == 7:
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        return t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3
    elif window_size_pos == 9:
        t_m4 = vectorized_token_using_pos_lookup(j-4, pos_tagged_sentence, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, pos_tagged_sentence, lookup_table_POS)
        return t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4
    elif window_size_pos == 11:
        t_m5 = vectorized_token_using_pos_lookup(j-5, pos_tagged_sentence, lookup_table_POS)
        t_m4 = vectorized_token_using_pos_lookup(j-4, pos_tagged_sentence, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, pos_tagged_sentence, lookup_table_POS)
        t_p5 = vectorized_token_using_pos_lookup(j+5, pos_tagged_sentence, lookup_table_POS)
        return t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5
    elif window_size_pos == 13:
        t_m6 = vectorized_token_using_pos_lookup(j-6, pos_tagged_sentence, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, pos_tagged_sentence, lookup_table_POS)
        t_m4 = vectorized_token_using_pos_lookup(j-4, pos_tagged_sentence, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, pos_tagged_sentence, lookup_table_POS)
        t_p5 = vectorized_token_using_pos_lookup(j+5, pos_tagged_sentence, lookup_table_POS)
        t_p6 = vectorized_token_using_pos_lookup(j+6, pos_tagged_sentence, lookup_table_POS)
        return t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6
    elif window_size_pos == 15:
        t_m7 = vectorized_token_using_pos_lookup(j-7, pos_tagged_sentence, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, pos_tagged_sentence, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, pos_tagged_sentence, lookup_table_POS)
        t_m4 = vectorized_token_using_pos_lookup(j-4, pos_tagged_sentence, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, pos_tagged_sentence, lookup_table_POS)
        t_p5 = vectorized_token_using_pos_lookup(j+5, pos_tagged_sentence, lookup_table_POS)
        t_p6 = vectorized_token_using_pos_lookup(j+6, pos_tagged_sentence, lookup_table_POS)
        t_p7 = vectorized_token_using_pos_lookup(j+7, pos_tagged_sentence, lookup_table_POS)
        return t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7
    elif window_size_pos == 17:
        t_m8 = vectorized_token_using_pos_lookup(j-8, pos_tagged_sentence, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, pos_tagged_sentence, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, pos_tagged_sentence, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, pos_tagged_sentence, lookup_table_POS)
        t_m4 = vectorized_token_using_pos_lookup(j-4, pos_tagged_sentence, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, pos_tagged_sentence, lookup_table_POS)
        t_p5 = vectorized_token_using_pos_lookup(j+5, pos_tagged_sentence, lookup_table_POS)
        t_p6 = vectorized_token_using_pos_lookup(j+6, pos_tagged_sentence, lookup_table_POS)
        t_p7 = vectorized_token_using_pos_lookup(j+7, pos_tagged_sentence, lookup_table_POS)
        t_p8 = vectorized_token_using_pos_lookup(j+8, pos_tagged_sentence, lookup_table_POS)
        return t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8
    elif window_size_pos == 19:
        t_m9 = vectorized_token_using_pos_lookup(j-9, pos_tagged_sentence, lookup_table_POS)
        t_m8 = vectorized_token_using_pos_lookup(j-8, pos_tagged_sentence, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, pos_tagged_sentence, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, pos_tagged_sentence, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, pos_tagged_sentence, lookup_table_POS)
        t_m4 = vectorized_token_using_pos_lookup(j-4, pos_tagged_sentence, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, pos_tagged_sentence, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, pos_tagged_sentence, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, pos_tagged_sentence, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, pos_tagged_sentence, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, pos_tagged_sentence, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, pos_tagged_sentence, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, pos_tagged_sentence, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, pos_tagged_sentence, lookup_table_POS)
        t_p5 = vectorized_token_using_pos_lookup(j+5, pos_tagged_sentence, lookup_table_POS)
        t_p6 = vectorized_token_using_pos_lookup(j+6, pos_tagged_sentence, lookup_table_POS)
        t_p7 = vectorized_token_using_pos_lookup(j+7, pos_tagged_sentence, lookup_table_POS)
        t_p8 = vectorized_token_using_pos_lookup(j+8, pos_tagged_sentence, lookup_table_POS)
        t_p9 = vectorized_token_using_pos_lookup(j+9, pos_tagged_sentence, lookup_table_POS)
        return t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9












############################################
""" Bag-of-n-grams feature extraction """
############################################

def add_1gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_1_gram, window_size_1gram):
    # for training data
    for i, sentence in enumerate(pre_X_train):
        for j, token in enumerate(sentence):
            FE_X_train[i][j] += feature_extraction_1gram(j, sentence, lookup_1_gram, window_size_1gram)
            this_feature_length = len(FE_X_train[i][j])
    # for testing data
    for i, sentence in enumerate(pre_X_test):
        for j, token in enumerate(sentence):
            FE_X_test[i][j] += feature_extraction_1gram(j, sentence, lookup_1_gram, window_size_1gram)
    print('>> 1gram feature vector dim = ', this_feature_length)
    return FE_X_train, FE_X_test

def add_2gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_2_gram):
    # for training data
    for i, sentence in enumerate(pre_X_train):
        for j, token in enumerate(sentence):
            step2_feature_vector = [] # [ 1g_t(t-1), 1g_t(t), 1g_t(t+1) ]
            """ [ token(t-1), token(t) ] """
            step2_feature_vector += numbering_token_using_2gram_lookup(j-1, j, sentence, lookup_2_gram)
            """ [ token(t), token(t-1) ] """
            step2_feature_vector += numbering_token_using_2gram_lookup(j, j+1, sentence, lookup_2_gram)
            FE_X_train[i][j] += step2_feature_vector # feature assignment
            this_feature_length = len(step2_feature_vector)
    # for testing data
    for i, sentence in enumerate(pre_X_test):
        for j, token in enumerate(sentence):
            step2_feature_vector = [] # [ 1g_t(t-1), 1g_t(t), 1g_t(t+1) ]
            """ [ token(t-1), token(t) ] """
            step2_feature_vector += numbering_token_using_2gram_lookup(j-1, j, sentence, lookup_2_gram)
            """ [ token(t), token(t-1) ] """
            step2_feature_vector += numbering_token_using_2gram_lookup(j, j+1, sentence, lookup_2_gram)
            FE_X_test[i][j] += step2_feature_vector # feature assignment
    print('>> 2gram feature vector dim (left+right-2gram) = ', this_feature_length)
    return FE_X_train, FE_X_test

def add_3gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_3_gram):
    # for training data
    for i, sentence in enumerate(pre_X_train):
        for j, token in enumerate(sentence):
            step3_feature_vector = [] # [ 1g_t(t-1), 1g_t(t), 1g_t(t+1) ]
            """ [ token(t-2), token(t-1), token(t) ] """
            step3_feature_vector += numbering_token_using_3gram_lookup(j-2, j-1, j, sentence, lookup_3_gram)
            """ [ token(t-1), token(t), token(t+1) ] """
            step3_feature_vector += numbering_token_using_3gram_lookup(j-1, j, j+1, sentence, lookup_3_gram)
            """ [ token(t), token(t+1), token(t+2) ] """
            step3_feature_vector += numbering_token_using_3gram_lookup(j, j+1, j+2, sentence, lookup_3_gram)
            FE_X_train[i][j] += step3_feature_vector # feature assignment
            this_feature_length = len(step3_feature_vector)
    # for testing data
    for i, sentence in enumerate(pre_X_test):
        for j, token in enumerate(sentence):
            step3_feature_vector = [] # [ 1g_t(t-1), 1g_t(t), 1g_t(t+1) ]
            """ [ token(t-2), token(t-1), token(t) ] """
            step3_feature_vector += numbering_token_using_3gram_lookup(j-2, j-1, j, sentence, lookup_3_gram)
            """ [ token(t-1), token(t), token(t+1) ] """
            step3_feature_vector += numbering_token_using_3gram_lookup(j-1, j, j+1, sentence, lookup_3_gram)
            """ [ token(t), token(t+1), token(t+2) ] """
            step3_feature_vector += numbering_token_using_3gram_lookup(j, j+1, j+2, sentence, lookup_3_gram)
            FE_X_test[i][j] += step3_feature_vector # feature assignment
    # Wrtie feature information
    print('>> 3gram feature vector dim = ', this_feature_length)
    return FE_X_train, FE_X_test

def add_5gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_5_gram):
    # for training data
    for i, sentence in enumerate(pre_X_train):
        for j, token in enumerate(sentence):
            step4_feature_vector = [] # [ 1g_t(t-1), 1g_t(t), 1g_t(t+1) ]
            """ [ token(t-2), token(t-1), token(t), token(t+1), token(t+2) ] """
            step4_feature_vector += numbering_token_using_5gram_lookup(j-2, j-1, j, j+1, j+2, sentence, lookup_5_gram)
            FE_X_train[i][j] += step4_feature_vector # feature assignment
            this_feature_length = len(step4_feature_vector)
    # for testing data
    for i, sentence in enumerate(pre_X_test):
        for j, token in enumerate(sentence):
            step4_feature_vector = [] # [ 1g_t(t-1), 1g_t(t), 1g_t(t+1) ]
            """ [ token(t-2), token(t-1), token(t), token(t+1), token(t+2) ] """
            step4_feature_vector += numbering_token_using_5gram_lookup(j-2, j-1, j, j+1, j+2, sentence, lookup_5_gram)
            FE_X_test[i][j] += step4_feature_vector # feature assignment
    # Wrtie feature information
    print('>> 5gram feature vector dim = ', this_feature_length)
    return FE_X_train, FE_X_test

def add_classes_1gram(FE_X_train, FE_X_test, pre_X_train, pre_X_test, lookup_classes_1gram, window_size_class):
    before_len = len(FE_X_train[0][0])
    # for training data
    for i, sentence in enumerate(pre_X_train):
        for j, token in enumerate(sentence):
            for lookup_per_class in lookup_classes_1gram:
                FE_X_train[i][j] += feature_extraction_1gram(j, sentence, lookup_per_class, window_size_class)
    # for testing data
    for i, sentence in enumerate(pre_X_test):
        for j, token in enumerate(sentence):
            for lookup_per_class in lookup_classes_1gram:
                FE_X_test[i][j] += feature_extraction_1gram(j, sentence, lookup_per_class, window_size_class)
    after_len = len(FE_X_train[0][0])
    print('>> lookup_classes_1gram feature vector dim = ', after_len-before_len)
    return FE_X_train, FE_X_test

############################################
""" Bag-of-n-grams feature design """
############################################

def numbering_token_using_1gram_lookup(j_plus_alpha, sentence, lookup, unknown=True):

    # token feature 0으로 초기화
    token_feature = [0] * len(lookup) # one-gram dictionary

    ## 1. for array index exception
    if j_plus_alpha < 0: # for first index exception
        return token_feature

    try:
        sentence[j_plus_alpha] # for last index exception
    except IndexError:
        return token_feature # 모든 element가 0인 token feature

#     print sentence[j_plus_alpha]
    ## 2. for key (unknown token) exception
    try:
        lookup[sentence[j_plus_alpha]]
    except KeyError:
        if unknown == True:
            token_feature[lookup['unknown_token']-1] = 1
            return token_feature # which is unknown index
        else: # unknown token 적용안할 것이다...
            return token_feature # 따라서, 그냥 0벡터로..

    ## 3. oridinal case
    token_feature[lookup[sentence[j_plus_alpha]]-1] = 1
    return token_feature

def numbering_token_using_2gram_lookup(j1, j2, sentence, lookup):

    # token feature 0으로 초기화
    token_feature = [0] * len(lookup) # bi-gram dictionary

    ## 1. for array index exception
    if j1 < 0: # for first index exception
        return token_feature

    try:
        sentence[j2] # for last index exception
    except IndexError:
        return token_feature # 모든 element가 0인 token feature

#     print sentence[j_plus_alpha]

    ## 2. for key (unknown token) exception
    tuple_2gram = (sentence[j1], sentence[j2])
    try:
        lookup[tuple_2gram]
    except KeyError:
        token_feature[lookup[('unknown_token', 'unknown_token')]-1] = 1
        return token_feature # which is unknown index

    ## 3. oridinal case
    token_feature[lookup[tuple_2gram]-1] = 1
    return token_feature

def numbering_token_using_3gram_lookup(j1, j2, j3, sentence, lookup):

    # token feature 0으로 초기화
    token_feature = [0] * len(lookup) # bi-gram dictionary

    ## 1. for array index exception
    if j1 < 0 or j2 < 0: # for first index exception
        return token_feature

    try:
        sentence[j2] # for last index exception
        sentence[j3]
    except IndexError:
        return token_feature # 모든 element가 0인 token feature

#     print sentence[j_plus_alpha]

    ## 2. for key (unknown token) exception
    tuple_3gram = (sentence[j1], sentence[j2], sentence[j3])
    try:
        lookup[tuple_3gram]
    except KeyError:
        token_feature[lookup[('unknown_token', 'unknown_token', 'unknown_token')]-1] = 1
        return token_feature # which is unknown index

    ## 3. oridinal case
    token_feature[lookup[tuple_3gram]-1] = 1
    return token_feature

def numbering_token_using_5gram_lookup(j1, j2, j3, j4, j5, sentence, lookup):

    # token feature 0으로 초기화
    token_feature = [0] * len(lookup) # bi-gram dictionary

    ## 1. for array index exception
    if j1 < 0 or j2 < 0: # for first index exception
        return token_feature

    try:
        sentence[j4] # for last index exception
        sentence[j5]
    except IndexError:
        return token_feature # 모든 element가 0인 token feature

#     print sentence[j_plus_alpha]

    ## 2. for key (unknown token) exception
    tuple_5gram = (sentence[j1], sentence[j2], sentence[j3], sentence[j4], sentence[j5])
    try:
        lookup[tuple_5gram]
    except KeyError:
        token_feature[lookup[('unknown_token', 'unknown_token', 'unknown_token', 'unknown_token', 'unknown_token')]-1] = 1
        return token_feature # which is unknown index

    ## 3. oridinal case
    token_feature[lookup[tuple_5gram]-1] = 1
    return token_feature

# def feature_extraction_1gram(j, sentence, word_one_gram_lookup_table):

#     t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
#     t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
#     current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
#     t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
#     t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)

# #     ## 1. 첫 번째 표현방법: 그냥 BIT OR연산으로 3개의 array를 하나로 통합한다.
# #     # convert to numpy and or bit calculation
# #     t_minus_one = np.array(t_minus_one)
# #     current_t = np.array(current_t)
# #     t_plus_one = np.array(t_plus_one)
# #     step1_feature_vector = t_minus_one | current_t | t_plus_one
# #     step1_feature_vector = np.array(step1_feature_vector).tolist()
# #     return step1_feature_vector # feature assignment

#     ## 2. 두 번째 표현방법: 그냥 병렬로 펼쳐버린다.
#     #return t_minus_one + current_t + t_plus_one
#     return t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two

def feature_extraction_1gram(j, sentence, word_one_gram_lookup_table, window_size):

    if window_size == 1:
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        return current_t
    elif window_size == 3:
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        return t_minus_one + current_t + t_plus_one
    elif window_size == 5:
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        return t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two
    elif window_size == 7:
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        return t_minus_three + t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three

    elif window_size == 9:
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        return t_minus_four + t_minus_three + t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four

    elif window_size == 11:
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        return t_minus_five + t_minus_four + t_minus_three + t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five

    elif window_size == 13:
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        return t_minus_six + t_minus_five + t_minus_four + t_minus_three + t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + t_plus_six

    elif window_size == 15:
        t_minus_seven = numbering_token_using_1gram_lookup(j-7, sentence, word_one_gram_lookup_table)
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        t_plus_seven = numbering_token_using_1gram_lookup(j+7, sentence, word_one_gram_lookup_table)
        return t_minus_seven + t_minus_six + t_minus_five + t_minus_four + t_minus_three + t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + t_plus_six + t_plus_seven

    elif window_size == 17:
        t_minus_eight = numbering_token_using_1gram_lookup(j-8, sentence, word_one_gram_lookup_table)
        t_minus_seven = numbering_token_using_1gram_lookup(j-7, sentence, word_one_gram_lookup_table)
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        t_plus_seven = numbering_token_using_1gram_lookup(j+7, sentence, word_one_gram_lookup_table)
        t_plus_eight = numbering_token_using_1gram_lookup(j+8, sentence, word_one_gram_lookup_table)
        return t_minus_eight + t_minus_seven + t_minus_six + t_minus_five + t_minus_four + t_minus_three + \
    t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + \
    t_plus_six + t_plus_seven + t_plus_eight

    elif window_size == 19:
        t_minus_nine = numbering_token_using_1gram_lookup(j-9, sentence, word_one_gram_lookup_table)
        t_minus_eight = numbering_token_using_1gram_lookup(j-8, sentence, word_one_gram_lookup_table)
        t_minus_seven = numbering_token_using_1gram_lookup(j-7, sentence, word_one_gram_lookup_table)
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        t_plus_seven = numbering_token_using_1gram_lookup(j+7, sentence, word_one_gram_lookup_table)
        t_plus_eight = numbering_token_using_1gram_lookup(j+8, sentence, word_one_gram_lookup_table)
        t_plus_nine = numbering_token_using_1gram_lookup(j+9, sentence, word_one_gram_lookup_table)
        return t_minus_nine + t_minus_eight + t_minus_seven + t_minus_six + t_minus_five + t_minus_four + t_minus_three + \
    t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + \
    t_plus_six + t_plus_seven + t_plus_eight + t_plus_nine

    elif window_size == 21:
        t_minus_ten = numbering_token_using_1gram_lookup(j-10, sentence, word_one_gram_lookup_table)
        t_minus_nine = numbering_token_using_1gram_lookup(j-9, sentence, word_one_gram_lookup_table)
        t_minus_eight = numbering_token_using_1gram_lookup(j-8, sentence, word_one_gram_lookup_table)
        t_minus_seven = numbering_token_using_1gram_lookup(j-7, sentence, word_one_gram_lookup_table)
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        t_plus_seven = numbering_token_using_1gram_lookup(j+7, sentence, word_one_gram_lookup_table)
        t_plus_eight = numbering_token_using_1gram_lookup(j+8, sentence, word_one_gram_lookup_table)
        t_plus_nine = numbering_token_using_1gram_lookup(j+9, sentence, word_one_gram_lookup_table)
        t_plus_ten = numbering_token_using_1gram_lookup(j+10, sentence, word_one_gram_lookup_table)
        return t_minus_ten + t_minus_nine + t_minus_eight + t_minus_seven + t_minus_six + t_minus_five + t_minus_four + t_minus_three + \
    t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + \
    t_plus_six + t_plus_seven + t_plus_eight + t_plus_nine + t_plus_ten

    elif window_size == 23:
        t_minus_eleven = numbering_token_using_1gram_lookup(j-11, sentence, word_one_gram_lookup_table)
        t_minus_ten = numbering_token_using_1gram_lookup(j-10, sentence, word_one_gram_lookup_table)
        t_minus_nine = numbering_token_using_1gram_lookup(j-9, sentence, word_one_gram_lookup_table)
        t_minus_eight = numbering_token_using_1gram_lookup(j-8, sentence, word_one_gram_lookup_table)
        t_minus_seven = numbering_token_using_1gram_lookup(j-7, sentence, word_one_gram_lookup_table)
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        t_plus_seven = numbering_token_using_1gram_lookup(j+7, sentence, word_one_gram_lookup_table)
        t_plus_eight = numbering_token_using_1gram_lookup(j+8, sentence, word_one_gram_lookup_table)
        t_plus_nine = numbering_token_using_1gram_lookup(j+9, sentence, word_one_gram_lookup_table)
        t_plus_ten = numbering_token_using_1gram_lookup(j+10, sentence, word_one_gram_lookup_table)
        t_plus_eleven = numbering_token_using_1gram_lookup(j+11, sentence, word_one_gram_lookup_table)
        return t_minus_eleven + t_minus_ten + t_minus_nine + t_minus_eight + t_minus_seven + t_minus_six + t_minus_five + t_minus_four + t_minus_three + \
    t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + \
    t_plus_six + t_plus_seven + t_plus_eight + t_plus_nine + t_plus_ten + t_plus_eleven

    elif window_size == 25:
        t_minus_twelve = numbering_token_using_1gram_lookup(j-12, sentence, word_one_gram_lookup_table)
        t_minus_eleven = numbering_token_using_1gram_lookup(j-11, sentence, word_one_gram_lookup_table)
        t_minus_ten = numbering_token_using_1gram_lookup(j-10, sentence, word_one_gram_lookup_table)
        t_minus_nine = numbering_token_using_1gram_lookup(j-9, sentence, word_one_gram_lookup_table)
        t_minus_eight = numbering_token_using_1gram_lookup(j-8, sentence, word_one_gram_lookup_table)
        t_minus_seven = numbering_token_using_1gram_lookup(j-7, sentence, word_one_gram_lookup_table)
        t_minus_six = numbering_token_using_1gram_lookup(j-6, sentence, word_one_gram_lookup_table)
        t_minus_five = numbering_token_using_1gram_lookup(j-5, sentence, word_one_gram_lookup_table)
        t_minus_four = numbering_token_using_1gram_lookup(j-4, sentence, word_one_gram_lookup_table)
        t_minus_three = numbering_token_using_1gram_lookup(j-3, sentence, word_one_gram_lookup_table)
        t_minus_two = numbering_token_using_1gram_lookup(j-2, sentence, word_one_gram_lookup_table)
        t_minus_one = numbering_token_using_1gram_lookup(j-1, sentence, word_one_gram_lookup_table)
        current_t = numbering_token_using_1gram_lookup(j, sentence, word_one_gram_lookup_table)
        t_plus_one = numbering_token_using_1gram_lookup(j+1, sentence, word_one_gram_lookup_table)
        t_plus_two = numbering_token_using_1gram_lookup(j+2, sentence, word_one_gram_lookup_table)
        t_plus_three = numbering_token_using_1gram_lookup(j+3, sentence, word_one_gram_lookup_table)
        t_plus_four = numbering_token_using_1gram_lookup(j+4, sentence, word_one_gram_lookup_table)
        t_plus_five = numbering_token_using_1gram_lookup(j+5, sentence, word_one_gram_lookup_table)
        t_plus_six = numbering_token_using_1gram_lookup(j+6, sentence, word_one_gram_lookup_table)
        t_plus_seven = numbering_token_using_1gram_lookup(j+7, sentence, word_one_gram_lookup_table)
        t_plus_eight = numbering_token_using_1gram_lookup(j+8, sentence, word_one_gram_lookup_table)
        t_plus_nine = numbering_token_using_1gram_lookup(j+9, sentence, word_one_gram_lookup_table)
        t_plus_ten = numbering_token_using_1gram_lookup(j+10, sentence, word_one_gram_lookup_table)
        t_plus_eleven = numbering_token_using_1gram_lookup(j+11, sentence, word_one_gram_lookup_table)
        t_plus_twelve = numbering_token_using_1gram_lookup(j+12, sentence, word_one_gram_lookup_table)
        return t_minus_twelve + t_minus_eleven + t_minus_ten + t_minus_nine + t_minus_eight + t_minus_seven + t_minus_six + t_minus_five + t_minus_four + t_minus_three + \
    t_minus_two + t_minus_one + current_t + t_plus_one + t_plus_two + t_plus_three + t_plus_four + t_plus_five + \
    t_plus_six + t_plus_seven + t_plus_eight + t_plus_nine + t_plus_ten + t_plus_eleven + t_plus_twelve

#     ## 1. 첫 번째 표현방법: 그냥 BIT OR연산으로 3개의 array를 하나로 통합한다.
#     # convert to numpy and or bit calculation
#     t_minus_one = np.array(t_minus_one)
#     current_t = np.array(current_t)
#     t_plus_one = np.array(t_plus_one)
#     step1_feature_vector = t_minus_one | current_t | t_plus_one
#     step1_feature_vector = np.array(step1_feature_vector).tolist()
#     return step1_feature_vector # feature assignment

    ## 2. 두 번째 표현방법: 그냥 병렬로 펼쳐버린다.
    #return t_minus_one + current_t + t_plus_one

def FE_for_classLUT(j, sentence):
    # 모든 class lookup table들을 이 함수를 통해 한번에 다 하자.
    # 그리고 unknown token은 추가하지 않는다. -> False
    # unknown token을 추가하니까 성능이 2%가량 향상되었다.
    unknown_token = True

    component  = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_component, unknown_token)
    refinement_of_component  = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_refinement_of_component, unknown_token)
    action  = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_action, unknown_token)
    refinement_of_action = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_refinement_of_action, unknown_token)
    condition = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_condition, unknown_token)
    priority = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_priority, unknown_token)
    motivation = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_motivation, unknown_token)
    role = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_role, unknown_token)
    object = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_object, unknown_token)
    refinement_of_object = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_refinement_of_object, unknown_token)
    sub_action = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_sub_action, unknown_token)
    sub_argument_of_action = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_sub_argument_of_action, unknown_token)
    sub_priority = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_sub_priority, unknown_token)
    sub_role = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_sub_role, unknown_token)
    sub_object = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_sub_object, unknown_token)
    sub_refinement_of_object = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_sub_refinement_of_object, unknown_token)
    none = numbering_token_using_1gram_lookup(j, sentence, lookup_table_class_none, unknown_token)
    return component + refinement_of_component + action + refinement_of_action + condition + priority + motivation + role + object + refinement_of_object + sub_action + sub_argument_of_action + sub_priority + sub_role + sub_object + sub_refinement_of_object + none


###################################################################
""" Load temp-data & lookup-tables and Init feature vectors"""
###################################################################

def init_feature_vec(pre_X_train, pre_X_test):
    FE2_X_train = []
    for sentence in (pre_X_train):
        temp1 = []
        for token in (sentence):
            temp1.append([])
        FE2_X_train.append(temp1)

    FE2_X_test = []
    for sentence in (pre_X_test):
        temp2 = []
        for token in (sentence):
            temp2.append([])
        FE2_X_test.append(temp2)
    return FE2_X_train, FE2_X_test

def load_temp_data(TEMP_DATA_PATH):
    pre_X_train = load(TEMP_DATA_PATH+'pre_X_train')
    pre_X_test = load(TEMP_DATA_PATH+'pre_X_test')
    Xtrain = load(TEMP_DATA_PATH+'Xtrain')
    Xtest = load(TEMP_DATA_PATH+'Xtest')
    Ytrain = load(TEMP_DATA_PATH+'Ytrain')
    Ytest = load(TEMP_DATA_PATH+'Ytest')
    return [pre_X_train, pre_X_test, Xtrain, Xtest, Ytrain, Ytest]

def load_lookup_tables(LOOKUP_TABLE_PATH):
    lookup_1_gram = load(LOOKUP_TABLE_PATH+'lookup_1_gram')
    lookup_2_gram = load(LOOKUP_TABLE_PATH+'lookup_2_gram')
    lookup_3_gram = load(LOOKUP_TABLE_PATH+'lookup_3_gram')
    lookup_5_gram = load(LOOKUP_TABLE_PATH+'lookup_5_gram')
    lookup_classes_1gram = load(LOOKUP_TABLE_PATH+'lookup_classes_1gram')
    return [lookup_1_gram, lookup_2_gram, lookup_3_gram, lookup_5_gram, lookup_classes_1gram]
