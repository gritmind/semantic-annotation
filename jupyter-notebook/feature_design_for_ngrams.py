


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

    

    
    
    
    
    
    