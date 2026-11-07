#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jaccard similarity
def jaccard(s1, s2):
    a = set(s1)
    b = set(s2)
    if len(a) == 0 or len(b) == 0:
        return 0
    return float(len(a & b) / len(a | b))


# Overlapping ngrams to create sets
def ngrams(seq, n):
    intermediate = [seq[i:i + n] for i in range(1 + len(seq) - n)]
    result = [i for i in intermediate if i]
    return result


# sliding window
def window(fseq, window):
    for i in range(len(fseq) - window + 1):
        yield fseq[i:i + window]


def normalized_similarity(ref1, ref2):
    number_of_grams = 1 # single gram tokenization
    ngram_sliding_window_size = 5 # single gram sliding window acts as a sampling approach to decreas computation
    n2 = ngrams(''.join(ref1), number_of_grams) # this converts the input text into a list of single/more grams or characters in the same order including empty characters
    # print(n2)
    s2 = {''.join(x) for x in window(n2, ngram_sliding_window_size)} # this slides over the list of characters creates a set of unique characters found in the list
    # print(s2)
    n1 = ngrams(''.join(ref2), number_of_grams)
    # print(n1)
    s1 = {''.join(x) for x in window(n1, ngram_sliding_window_size)}
    # print(s1)
    score = jaccard(s1, s2) #this find the jaccard coefficient between the two sets
    return score


if __name__ == '__main__':
    # {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A985464', 'reference': 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106'}
    # {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A956423', 'reference': 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456 18 2098'}
    # {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A956296', 'reference': 'LLOYD AARON D 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456 18 2098'}
    # {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A830349', 'reference': 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456182098'}
    # {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A875214', 'reference': 'ANDREEW AARON STEPHEN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 601 70 6106'}
    # {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A824917', 'reference': 'ANDREW AARON STEPHEN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 601 70 6106'}
    ref1 = ['LLOYD', 'AARON', 'DEAN', '2475', 'SPICEWOOD', 'DR', 'WINSTON', 'SALEM', 'NC', '27106', '456', '18', '2098']
    ref2 = ['LLOYD', 'AARON', 'DEAN', '2475', 'SPICEWOOD', 'DR', 'WINSTON', 'SALEM', 'NC', '27106', '456', '18', '2098', '3/']

    # ref1 = ['L LOYD']
    # ref2 = ['LOYD']
    print(ref1)
    print(str(''.join(ref1)))
    s = normalized_similarity(ref1, ref2)
    # s = DWM65_ScoringMatrixStd.normalized_similarity(ref1, ref2)
    print(s)
