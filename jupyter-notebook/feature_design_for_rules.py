

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














    