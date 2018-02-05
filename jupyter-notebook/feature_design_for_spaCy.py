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

    
    
######################
""" Part Of Speech """
######################

def what_is_headPOS(token, lookup):
    
    token_feature = [0] * len(lookup)
    token_feature[lookup[token.head.pos_]] = 1
    return token_feature


def vectorized_token_using_pos_lookup(j_plus_alpha, spacy_sent, lookup):
    
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
        t__0 = vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        return t__0
    elif window_size_pos == 3:
        t_m1 = vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)        
        return t_m1 + t__0 + t_p1
    elif window_size_pos == 5:
        t_m2 = vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        return t_m2 + t_m1 + t__0 + t_p1 + t_p2    
    elif window_size_pos == 7:
        t_m3 = vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        return t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3     
    elif window_size_pos == 9:
        t_m4 = vectorized_token_using_pos_lookup(j-4, spacy_sent, lookup_table_POS)
        t_m3 = vectorized_token_using_pos_lookup(j-3, spacy_sent, lookup_table_POS)
        t_m2 = vectorized_token_using_pos_lookup(j-2, spacy_sent, lookup_table_POS)
        t_m1 = vectorized_token_using_pos_lookup(j-1, spacy_sent, lookup_table_POS)
        t__0 = vectorized_token_using_pos_lookup(j, spacy_sent, lookup_table_POS)
        t_p1 = vectorized_token_using_pos_lookup(j+1, spacy_sent, lookup_table_POS)
        t_p2 = vectorized_token_using_pos_lookup(j+2, spacy_sent, lookup_table_POS)
        t_p3 = vectorized_token_using_pos_lookup(j+3, spacy_sent, lookup_table_POS)
        t_p4 = vectorized_token_using_pos_lookup(j+4, spacy_sent, lookup_table_POS)
        return t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4     
    elif window_size_pos == 11:
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        return t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5     
    elif window_size_pos == 13:
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        return t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6
    elif window_size_pos == 15:
        t_m7 = vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
    elif window_size_pos == 17:
        t_m8 = vectorized_token_using_pos_lookup(j-8, spacy_sent, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        t_p8 = vectorized_token_using_pos_lookup(j+8, spacy_sent, lookup_table_POS)
        return t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8    
    elif window_size_pos == 19:
        t_m9 = vectorized_token_using_pos_lookup(j-9, spacy_sent, lookup_table_POS)
        t_m8 = vectorized_token_using_pos_lookup(j-8, spacy_sent, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        t_p8 = vectorized_token_using_pos_lookup(j+8, spacy_sent, lookup_table_POS)
        t_p9 = vectorized_token_using_pos_lookup(j+9, spacy_sent, lookup_table_POS)
        return t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 
    elif window_size_pos == 21:
        t_m10 = vectorized_token_using_pos_lookup(j-10, spacy_sent, lookup_table_POS)
        t_m9 = vectorized_token_using_pos_lookup(j-9, spacy_sent, lookup_table_POS)
        t_m8 = vectorized_token_using_pos_lookup(j-8, spacy_sent, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        t_p8 = vectorized_token_using_pos_lookup(j+8, spacy_sent, lookup_table_POS)
        t_p9 = vectorized_token_using_pos_lookup(j+9, spacy_sent, lookup_table_POS)
        t_p10 = vectorized_token_using_pos_lookup(j+10, spacy_sent, lookup_table_POS)
        return t_m10 + t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 + t_p10
    elif window_size_pos == 23:
        t_m11 = vectorized_token_using_pos_lookup(j-11, spacy_sent, lookup_table_POS)
        t_m10 = vectorized_token_using_pos_lookup(j-10, spacy_sent, lookup_table_POS)
        t_m9 = vectorized_token_using_pos_lookup(j-9, spacy_sent, lookup_table_POS)
        t_m8 = vectorized_token_using_pos_lookup(j-8, spacy_sent, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        t_p8 = vectorized_token_using_pos_lookup(j+8, spacy_sent, lookup_table_POS)
        t_p9 = vectorized_token_using_pos_lookup(j+9, spacy_sent, lookup_table_POS)
        t_p10 = vectorized_token_using_pos_lookup(j+10, spacy_sent, lookup_table_POS)
        t_p11 = vectorized_token_using_pos_lookup(j+11, spacy_sent, lookup_table_POS)
        return t_m11 + t_m10 + t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 + t_p10 + t_p11
    elif window_size_pos == 25:
        t_m12 = vectorized_token_using_pos_lookup(j-12, spacy_sent, lookup_table_POS)
        t_m11 = vectorized_token_using_pos_lookup(j-11, spacy_sent, lookup_table_POS)
        t_m10 = vectorized_token_using_pos_lookup(j-10, spacy_sent, lookup_table_POS)
        t_m9 = vectorized_token_using_pos_lookup(j-9, spacy_sent, lookup_table_POS)
        t_m8 = vectorized_token_using_pos_lookup(j-8, spacy_sent, lookup_table_POS)
        t_m7 = vectorized_token_using_pos_lookup(j-7, spacy_sent, lookup_table_POS)
        t_m6 = vectorized_token_using_pos_lookup(j-6, spacy_sent, lookup_table_POS)
        t_m5 = vectorized_token_using_pos_lookup(j-5, spacy_sent, lookup_table_POS)
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
        t_p8 = vectorized_token_using_pos_lookup(j+8, spacy_sent, lookup_table_POS)
        t_p9 = vectorized_token_using_pos_lookup(j+9, spacy_sent, lookup_table_POS)
        t_p10 = vectorized_token_using_pos_lookup(j+10, spacy_sent, lookup_table_POS)
        t_p11 = vectorized_token_using_pos_lookup(j+11, spacy_sent, lookup_table_POS)
        t_p12 = vectorized_token_using_pos_lookup(j+12, spacy_sent, lookup_table_POS)
        return t_m12 + t_m11 + t_m10 + t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 + t_p10 + t_p11 + t_p12




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

		
		
		
		
		
####################
""" Full Parsing """
####################

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
		


        


##########################
""" Dependency Parsing """
##########################


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
    
    
    
    
#def distance_toRoot_token_based(j, sentence):











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
