import nltk
import collections

######################
""" Part Of Speech """
######################

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
    elif window_size_pos == 21:
        t_m10 = vectorized_token_using_pos_lookup(j-10, pos_tagged_sentence, lookup_table_POS)
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
        t_p10 = vectorized_token_using_pos_lookup(j+10, pos_tagged_sentence, lookup_table_POS)
        return t_m10 + t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 + t_p10
    elif window_size_pos == 23:
        t_m11 = vectorized_token_using_pos_lookup(j-11, pos_tagged_sentence, lookup_table_POS)
        t_m10 = vectorized_token_using_pos_lookup(j-10, pos_tagged_sentence, lookup_table_POS)
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
        t_p10 = vectorized_token_using_pos_lookup(j+10, pos_tagged_sentence, lookup_table_POS)
        t_p11 = vectorized_token_using_pos_lookup(j+11, pos_tagged_sentence, lookup_table_POS)
        return t_m11 + t_m10 + t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 + t_p10 + t_p11
    elif window_size_pos == 25:
        t_m12 = vectorized_token_using_pos_lookup(j-12, pos_tagged_sentence, lookup_table_POS)
        t_m11 = vectorized_token_using_pos_lookup(j-11, pos_tagged_sentence, lookup_table_POS)
        t_m10 = vectorized_token_using_pos_lookup(j-10, pos_tagged_sentence, lookup_table_POS)
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
        t_p10 = vectorized_token_using_pos_lookup(j+10, pos_tagged_sentence, lookup_table_POS)
        t_p11 = vectorized_token_using_pos_lookup(j+11, pos_tagged_sentence, lookup_table_POS)
        t_p12 = vectorized_token_using_pos_lookup(j+12, pos_tagged_sentence, lookup_table_POS)
        return t_m12 + t_m11 + t_m10 + t_m9 + t_m8 + t_m7 + t_m6 + t_m5 + t_m4 + t_m3 + t_m2 + t_m1 + t__0 + \
               t_p1 + t_p2 + t_p3 + t_p4 + t_p5 + t_p6 + t_p7 + t_p8 + t_p9 + t_p10 + t_p11 + t_p12
#     token_feature_left = [0] * len(lookup_table_POS)
#     token_feature_center = [0] * len(lookup_table_POS)
#     token_feature_right = [0] * len(lookup_table_POS)    
    
#     if window_size_pos == 1:
#         token_feature_center[lookup_table_POS[pos_tagged_sentence[j][1]]] = 1
#         return token_feature_center
    
#     elif window_size_pos == 3:
#         if j > 0 and j <len(pos_tagged_sentence)-1:
#             token_feature_center[lookup_table_POS[pos_tagged_sentence[j][1]]] = 1
#             token_feature_left[lookup_table_POS[pos_tagged_sentence[j-1][1]]] = 1
#             token_feature_right[lookup_table_POS[pos_tagged_sentence[j+1][1]]] = 1
#         elif j==0:
#             token_feature_center[lookup_table_POS[pos_tagged_sentence[j][1]]] = 1
#             token_feature_right[lookup_table_POS[pos_tagged_sentence[j+1][1]]] = 1
#         elif j==len(pos_tagged_sentence)-1:
#             token_feature_left[lookup_table_POS[pos_tagged_sentence[j-1][1]]] = 1
#             token_feature_center[lookup_table_POS[pos_tagged_sentence[j][1]]] = 1
#         return token_feature_left + token_feature_center + token_feature_right






##################################
""" Chunking (Shallow Parsing) """
##################################
def FE_NP_CHUNK(j, sentence, NP_word_list, window_size_np_chunk):
    token_feature_left = [0]
    token_feature_center = [0]
    token_feature_right = [0]
    
   
    
    if window_size_np_chunk == 1:
        if any(sentence[j] in t for t in NP_word_list):
            token_feature_center[0] = 1 # NP True
            return token_feature_center
        else:
            return token_feature_center # NP False
    
    elif window_size_np_chunk == 3:
       
        if j > 0 and j < len(sentence)-1:
            
            if any(sentence[j-1] in t for t in NP_word_list):
                token_feature_left[0] = 1 # NP True            
            if any(sentence[j] in t for t in NP_word_list):
                token_feature_center[0] = 1 # NP True
            if any(sentence[j+1] in t for t in NP_word_list):
                token_feature_right[0] = 1 # NP True            
        elif j==0:
            if any(sentence[j] in t for t in NP_word_list):
                token_feature_center[0] = 1
            if any(sentence[j+1] in t for t in NP_word_list):
                token_feature_right[0] = 1
        elif j==len(sentence)-1:
            if any(sentence[j] in t for t in NP_word_list):
                token_feature_left[0] = 1
            if any(sentence[j-1] in t for t in NP_word_list):
                token_feature_center[0] = 1
        
        return token_feature_left + token_feature_center + token_feature_right 
		



		
		
		
		
		
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
