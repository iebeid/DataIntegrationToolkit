import re

import numpy as np
import modules.DWM95_CalculateEntropy
import modules.DWM65_ScoringMatrixStd as DWM65_ScoringMatrixStd
import modules.DWM16_BuildTokenFreqDict as DWM16_BuildTokenFreqDict
import textdistance
import Levenshtein


# -------------JACCARD--------------------
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


# ------------------------------------------

def a_record_centric_average_token_count(reference):
    token_counts = dict(zip(reference, [reference.count(i) for i in reference]))
    return np.mean(np.array(list(token_counts.values())))


def b_record_centric_max_token_count(reference):
    token_counts = dict(zip(reference, [reference.count(i) for i in reference]))
    return max(list(token_counts.values()))


def c_record_centric_min_token_count(reference):
    token_counts = dict(zip(reference, [reference.count(i) for i in reference]))
    return min(list(token_counts.values()))


def d_record_centric_average_token_length(reference):
    token_lengths = dict(zip(reference, [len(i) for i in reference]))
    return np.mean(np.array(list(token_lengths.values())))


def e_record_centric_max_token_length(reference):
    token_lengths = dict(zip(reference, [len(i) for i in reference]))
    return max(list(token_lengths.values()))


def f_record_centric_min_token_length(reference):
    token_lengths = dict(zip(reference, [len(i) for i in reference]))
    return min(list(token_lengths.values()))


def g_record_centric_number_of_tokens(reference):
    return len(reference)


def h_record_centric_number_of_unique_tokens(reference):
    return len(list(set(reference)))


def i_record_centric_number_of_characters(reference):
    return len(str(''.join(reference)))


def j_record_centric_number_of_unique_characters(reference):
    return len(list(set(list(str(''.join(reference))))))


def k_record_centric_number_of_digits(reference):
    return len(re.findall('\d', str(''.join(reference))))


def l_record_centric_number_of_unique_digits(reference):
    return len(list(set(re.findall('\d', str(''.join(reference))))))


def m_record_centric_number_of_alphabetical_chars(reference):
    return len(re.findall('[A-Za-z]', str(''.join(reference))))


def n_record_centric_number_of_unique_alphabetical_chars(reference):
    return len(list(set(re.findall('[A-Za-z]', str(''.join(reference))))))


def o_relation_centric_number_of_shared_tokens(reference1, reference2):
    return len(list(set.intersection(set(reference1), set(reference2)))) / len(list(set(reference1 + reference2)))


def p_relation_centric_number_of_shared_characters(reference1, reference2):
    return len(list(set.intersection(set(list(str(''.join(reference1)))),set(list(str(''.join(reference2))))))) / len(list(set(list(str(''.join(reference1))) + list(str(''.join(reference2))))))


def q_relation_centric_number_of_shared_digits(reference1, reference2):
    d_ref_1 = list(re.findall('\d', str(''.join(reference1))))
    d_ref_2 = list(re.findall('\d', str(''.join(reference2))))
    n_unique_digits = len(list(set(d_ref_1 + d_ref_2)))
    s = 0.0
    if len(d_ref_1) != 0 and len(d_ref_2) != 0:
        s = len(list(set.intersection(set(d_ref_1),set(d_ref_2)))) / n_unique_digits
    return s


def r_relation_centric_number_of_shared_alphabetical_chars(reference1, reference2):
    d_ref_1 = list(re.findall('[A-Za-z]', str(''.join(reference1))))
    d_ref_2 = list(re.findall('[A-Za-z]', str(''.join(reference2))))
    n_unique_digits = len(list(set(d_ref_1 + d_ref_2)))
    s = 0.0
    if len(d_ref_1) != 0 and len(d_ref_2) != 0:
        s = len(list(set.intersection(set(d_ref_1),set(d_ref_2)))) / n_unique_digits
    return s


def s_relation_centric_jaccard(reference1, reference2):
    number_of_grams = 2  # single gram tokenization
    ngram_sliding_window_size = 5  # single gram sliding window acts as a sampling approach to decreas computation
    n2 = ngrams(''.join(reference1),
                number_of_grams)  # this converts the input text into a list of single/more grams or characters in the same order including empty characters
    s2 = {''.join(x) for x in window(n2,
                                     ngram_sliding_window_size)}  # this slides over the list of characters creates a set of unique characters found in the list
    n1 = ngrams(''.join(reference2), number_of_grams)
    s1 = {''.join(x) for x in window(n1, ngram_sliding_window_size)}
    score = jaccard(s1, s2)  # this find the jaccard coefficient between the two sets
    return score

def character_jaccard(t1, t2):
    number_of_grams = 1  # single gram tokenization
    ngram_sliding_window_size = 1  # single gram sliding window acts as a sampling approach to decreas computation
    n2 = ngrams(''.join(t1),
                number_of_grams)  # this converts the input text into a list of single/more grams or characters in the same order including empty characters
    s2 = {''.join(x) for x in window(n2,
                                     ngram_sliding_window_size)}  # this slides over the list of characters creates a set of unique characters found in the list
    n1 = ngrams(''.join(t2), number_of_grams)
    s1 = {''.join(x) for x in window(n1, ngram_sliding_window_size)}
    score = jaccard(s1, s2)  # this find the jaccard coefficient between the two sets
    return score


def t_relation_centric_token_based_average_jaccard(reference1, reference2):
    sim_scores = []
    for t1 in reference1:
        token_similarity_scores = []
        for t2 in reference2:
            similarity_score = character_jaccard(t1, t2)
            token_similarity_scores.append(similarity_score)
        max_sim_score = float(max(token_similarity_scores))
        sim_scores.append(max_sim_score)
    return np.mean(np.array(sim_scores))


def u_relation_centric_token_shared_positions(reference1, reference2):
    refs_min = None
    refs_max = None
    if len(reference1) < len(reference2):
        refs_min = reference1
        refs_max = reference2
    if len(reference2) < len(reference1):
        refs_min = reference2
        refs_max = reference1
    if len(reference2) == len(reference1):
        refs_min = reference1
        refs_max = reference2
    count_true = 0
    for i, t in enumerate(refs_min):
        if t == refs_max[i]:
            count_true = count_true + 1
    return float(count_true / len(refs_max))





def similarity_probability(reference1, reference2):
    s = []

    s1 = a_record_centric_average_token_count(reference1)
    s2 = a_record_centric_average_token_count(reference2)
    if s1 > 0 and s2 > 0:
        s_a = min([s1, s2]) / max([s1, s2])
    else:
        s_a = 0.0
    print("a_record_centric_average_token_count")
    print(s_a)
    s.append(s_a)

    s1 = b_record_centric_max_token_count(reference1)
    s2 = b_record_centric_max_token_count(reference2)
    if s1 > 0 and s2 > 0:
        s_b = min([s1, s2]) / max([s1, s2])
    else:
        s_b = 0.0
    print("b_record_centric_max_token_count")
    print(s_b)
    s.append(s_b)

    s1 = c_record_centric_min_token_count(reference1)
    s2 = c_record_centric_min_token_count(reference2)
    if s1 > 0 and s2 > 0:
        s_c = min([s1, s2]) / max([s1, s2])
    else:
        s_c = 0.0
    print("c_record_centric_min_token_count")
    print(s_c)
    s.append(s_c)

    s1 = d_record_centric_average_token_length(reference1)
    s2 = d_record_centric_average_token_length(reference2)
    if s1 > 0 and s2 > 0:
        s_d = min([s1, s2]) / max([s1, s2])
    else:
        s_d = 0.0
    print("d_record_centric_average_token_length")
    print(s_d)
    s.append(s_d)

    s1 = e_record_centric_max_token_length(reference1)
    s2 = e_record_centric_max_token_length(reference2)
    if s1 > 0 and s2 > 0:
        s_e = min([s1, s2]) / max([s1, s2])
    else:
        s_e = 0.0
    print("e_record_centric_max_token_length")
    print(s_e)
    s.append(s_e)

    s1 = f_record_centric_min_token_length(reference1)
    s2 = f_record_centric_min_token_length(reference2)
    if s1 > 0 and s2 > 0:
        s_f = min([s1, s2]) / max([s1, s2])
    else:
        s_f = 0.0
    print("f_record_centric_number_of_characters")
    print(s_f)
    s.append(s_f)

    s1 = g_record_centric_number_of_tokens(reference1)
    s2 = g_record_centric_number_of_tokens(reference2)
    if s1 > 0 and s2 > 0:
        s_g = min([s1, s2]) / max([s1, s2])
    else:
        s_g = 0.0
    print("g_record_centric_number_of_tokens")
    print(s_g)
    s.append(s_g)

    s1 = h_record_centric_number_of_unique_tokens(reference1)
    s2 = h_record_centric_number_of_unique_tokens(reference2)
    if s1 > 0 and s2 > 0:
        s_h = min([s1, s2]) / max([s1, s2])
    else:
        s_h = 0.0
    print("h_record_centric_number_of_unique_tokens")
    print(s_h)
    s.append(s_h)

    s1 = i_record_centric_number_of_characters(reference1)
    s2 = i_record_centric_number_of_characters(reference2)
    if s1 > 0 and s2 > 0:
        s_i = min([s1, s2]) / max([s1, s2])
    else:
        s_i = 0.0
    print("i_record_centric_number_of_characters")
    print(s_i)
    s.append(s_i)

    s1 = j_record_centric_number_of_unique_characters(reference1)
    s2 = j_record_centric_number_of_unique_characters(reference2)
    if s1 > 0 and s2 > 0:
        s_j = min([s1, s2]) / max([s1, s2])
    else:
        s_j = 0.0
    print("j_record_centric_number_of_unique_characters")
    print(s_j)
    s.append(s_j)

    s1 = k_record_centric_number_of_digits(reference1)
    s2 = k_record_centric_number_of_digits(reference2)
    if s1 > 0 and s2 > 0:
        s_k = min([s1, s2]) / max([s1, s2])
    else:
        s_k = 0.0
    print("k_record_centric_number_of_digits")
    print(s_k)
    s.append(s_k)

    s1 = l_record_centric_number_of_unique_digits(reference1)
    s2 = l_record_centric_number_of_unique_digits(reference2)
    if s1 > 0 and s2 > 0:
        s_l = min([s1, s2]) / max([s1, s2])
    else:
        s_l = 0.0
    print("l_record_centric_number_of_unique_digits")
    print(s_l)
    s.append(s_l)

    s1 = m_record_centric_number_of_alphabetical_chars(reference1)
    s2 = m_record_centric_number_of_alphabetical_chars(reference2)
    if s1 > 0 and s2 > 0:
        s_m = min([s1, s2]) / max([s1, s2])
    else:
        s_m = 0.0
    print("m_record_centric_number_of_unique_special_chars")
    print(s_m)
    s.append(s_m)

    s1 = n_record_centric_number_of_unique_alphabetical_chars(reference1)
    s2 = n_record_centric_number_of_unique_alphabetical_chars(reference2)
    if s1 > 0 and s2 > 0:
        s_n = min([s1, s2]) / max([s1, s2])
    else:
        s_n = 0.0
    print("n_record_centric_number_of_unique_alphabetical_chars")
    print(s_n)
    s.append(s_n)

    s_o = o_relation_centric_number_of_shared_tokens(reference1, reference2)
    print("o_relation_centric_number_of_shared_tokens")
    print(s_o)
    s.append(s_o)

    s_p = p_relation_centric_number_of_shared_characters(reference1, reference2)
    print("p_relation_centric_number_of_shared_characters")
    print(s_p)
    s.append(s_p)

    s_q = q_relation_centric_number_of_shared_digits(reference1, reference2)
    print("q_relation_centric_number_of_shared_digits")
    print(s_q)
    s.append(s_q)

    s_r = r_relation_centric_number_of_shared_alphabetical_chars(reference1, reference2)
    print("r_relation_centric_number_of_shared_alphabetical_chars")
    print(s_r)
    s.append(s_r)

    s_s = u_relation_centric_token_shared_positions(reference1, reference2)
    print("u_relation_centric_token_shared_positions")
    print(s_s)
    s.append(s_s)

    # s_s = s_relation_centric_jaccard(reference1, reference2)
    # print("s_relation_centric_jaccard")
    # print(s_s)
    # s.append(s_s)

    # s_t = t_relation_centric_token_based_average_jaccard(reference1, reference2)
    # print("t_relation_centric_token_based_average_jaccard")
    # print(s_t)
    # s.append(s_t)

    # s_v = modules.DWM95_CalculateEntropy.calculateEntropy([reference1,reference2])
    # print("calculateEntropy")
    # print(s_v)
    # s.append(s_v)

    final_p = 1
    for p in s:
        if p != 0:
            final_p = final_p * p


    print("MIN/MAX")
    print(min(s)/max(s))
    print("MIN+MAX/2")
    print((min(s)+max(s))/2)
    print("AVERAGE of P")
    print(np.mean(np.array(s)))
    print("AVERAGE of P / MAX")
    print(np.mean(np.array(s))/max(s))
    print("MULTIPLIED")
    print(final_p)
    # print("Token JACCARD * ENTROPY")
    # print(s_t * s_v)
    print("Scoring Matrix")
    print(DWM65_ScoringMatrixStd.normalized_similarity(reference1,reference2))

    return final_p




# ref1 = ['LLOYD', 'AARON', 'DEAN', '2475', 'SPICEWOOD', 'DR', 'WINSTON', 'SALEM', 'NC', '27106', '456', '18', '2098']
# ref2 = ['LLOYD', 'AARON', 'DEAN', '2475', 'SPICEWOOD', 'DR', 'WINSTON', 'SALEM', 'NC', '27106', '456', '18', '2098',
#         '3/']
# {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A830349', 'reference': 'ANDREW AARON STEPHEN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 601 70 6106'}
# {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A875214', 'reference': 'ANDREEW AARON STEPHEN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 601 70 6106'}
# ref1 = 'LLOYD AARON DEAN SPICEWOOD DR WINSTON SALEM NC'.split()

# {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A985464', 'reference': 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106'}
# {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A956423', 'reference': 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456 18 2098'}
# {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A956296', 'reference': 'LLOYD AARON D 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456 18 2098'}
# {'cluster_id': 'A824917', 'quality': 0.8161164071642083, 'ref_id': 'A830349', 'reference': 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456182098'}

# ref1 = " 2475 SPICEWOOD DR WINSTON AARON SALEM NC 27106 STEPHEN 601 70 6106 ANDREEW".split()
# ref1 = 'LLOYD AARON DEAN SPICEWOOD DR WINSTON SALEM NC'.split()
# ref2 = 'D SPICEWOOD LLOYD DR WINSTON  AARON NC SALEM'.split()
# ref1 = ["LLOYD"]
# ref2 = ["D"]
# ref1 = 'BOY GIRL'.split()


def tokenizerCompress(string):
    string = string.upper()
    string = string.replace(",", ' ')
    tokenList = re.split('[\s]+', string)
    newList = []
    for token in tokenList:
        newToken = re.sub('[\W]+', '', token)
        if len(newToken) > 0:
            newList.append(newToken)
    return newList

# ***********Inner Function*******************************
# Replace all non-words characters with blanks, then split on blanks
def tokenizerSplitter(string):
    string = string.upper()
    string = re.sub('[\W]+', ' ', string)
    tokenList = re.split('[\s]+', string)
    newList = []
    for token in tokenList:
        if len(token) > 0:
            newList.append(token)
    return newList

# def weighted_comparator(ref1, ref2):



from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
def matching_probablity():
    # ref1 = 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456182098'.split()
    # ref1 = 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106'.split()
    # ref1 = "LLOYD AARON D 2475 SPICEWOOD DR WINSTON SALEM NC 27106 456 18 2098".split()
    # ref1 = 'ANDREEW AARON STEPHEN 2475 SPICEWOOD DR WINSTON SALEM NC 27106 601 70'.split()
    # ref2 = 'LLOYD AARON DEAN 2475 SPICEWOOD DR WINSTON SALEM NC 27106'.split()

    # ref1 = tokenizerSplitter("TRACEY AARON LEIGH 176 NOTTIDGE CT CLEMMONS NC 27012 954451341")
    # ref2 = tokenizerSplitter("MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012 642 19 4591")

    # ref1 = tokenizerCompress("TRACEY AARON LEIGH 176 NOTTIDGE CT CLEMMONS NC 27012 954451341")
    # # ref1 = tokenizerCompress("MICHAEL AARON SCOT 1010 PALISADE CT MARTINEZ CA 94553 642194591")
    # # ref2 = tokenizerCompress("MICHAEL AARON SCOT 1010 PALISADE CT MARTINEZ CA 94553 642194591")
    # ref2 = tokenizerCompress("MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012 642-19-4591")

    # ref1 = tokenizerCompress("TRACEY LEIGH 954451341")
    ref1 = tokenizerCompress("1010 PALISADE MARTINEZ CA 94553")
    ref2 = tokenizerCompress("176 NOTTIDGE CLEMMONS NC 27012")

    # "A905248,MICHAEL,AARON,SCOT,1010  PALISADE CT   ,MARTINEZ,CA,94553,642194591
    # "A950657,MICHAEL,AARON,SCOT,176 NOTTIDGE CT,CLEMMONS,NC,27012,642-19-4591"


    # ref1 = "TRACEY AARON LEIGH 176 NOTTIDGE CT CLEMMONS NC 27012 954451341".split()
    # ref2 = "MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012 642 19 4591".split()
    print(ref1)
    print(ref2)
    ref1_digit_tokens = []
    ref1_long_alphabetical_tokens = []
    ref1_short_alphabetical_tokens = []
    for t1 in ref1:
        if t1.isdigit():
            ref1_digit_tokens.append(t1)
        elif len(t1) < 5:
            ref1_short_alphabetical_tokens.append(t1)
        else:
            ref1_long_alphabetical_tokens.append(t1)
    ref2_digit_tokens = []
    ref2_long_alphabetical_tokens = []
    ref2_short_alphabetical_tokens = []
    for t2 in ref2:
        if t2.isdigit():
            ref2_digit_tokens.append(t2)
        elif len(t2) < 5:
            ref2_short_alphabetical_tokens.append(t2)
        else:
            ref2_long_alphabetical_tokens.append(t2)
    long_alphabetical_jaccard = s_relation_centric_jaccard(ref1_long_alphabetical_tokens,ref2_long_alphabetical_tokens)
    short_alphabetical_jaccard = s_relation_centric_jaccard(ref1_short_alphabetical_tokens, ref2_short_alphabetical_tokens)
    numerical_jaccard = s_relation_centric_jaccard(ref1_digit_tokens, ref2_digit_tokens)
    print(str(s_relation_centric_jaccard(ref1,ref2)))
    print(str(((1 * short_alphabetical_jaccard) + (2 * long_alphabetical_jaccard) + (8 * numerical_jaccard))/(1+2+8)))
    print("Scoring Matrix")
    print(DWM65_ScoringMatrixStd.normalized_similarity(ref1,ref2))
    print("Cosine Similairty")
    print(textdistance.cosine(ref1,ref2))
    print("TFIDF Similairty")
    corpus = []
    corpus.append(' '.join(ref1))
    corpus.append(' '.join(ref2))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_sim)

    p = DWM65_ScoringMatrixStd.normalized_similarity("ARON", "ARON")
    p = textdistance.cosine.normalized_similarity("ARON", "ARON")
    p = textdistance.jaccard.normalized_similarity("ARON", "ARON")
    p = textdistance.monge_elkan.normalized_similarity("ARON", "ARON")
    p = textdistance.bag.normalized_similarity("ARON", "ARON")
    p = textdistance.hamming.normalized_similarity("ARON", "ARON")
    p = textdistance.levenshtein.normalized_similarity("ARON", "ARON")
    p = textdistance.damerau_levenshtein.normalized_similarity("ARON", "ARON")

    S = None
    if ref1 == ref2 or set.issubset(set(ref1), set(ref2)) or set.issubset(set(ref2), set(ref1)):
        S = 1.0
    elif set.isdisjoint(set(ref1), set(ref2)):
        S = 0.0
    else:
        S = similarity_probability(ref1, ref2)
    print("P MULTIPLIED")
    print(S)

    # return S
# print(S)
# matching_probablity()
# s = modules.DWM65_ScoringMatrixStd.normalized_similarity(ref1,ref2)
# cluster=[]
# cluster.append(ref1)
# cluster.append(ref2)
# e = modules.DWM95_CalculateEntropy.calculateEntropy(cluster)
# # print(s)
# print(e)
from difflib import SequenceMatcher
Str1 = "L"
Str2 = "LACHELLE"
Distance = Levenshtein.distance(Str1.lower(),Str2.lower()),
print(Distance)
Ratio = Levenshtein.ratio(Str1.lower(),Str2.lower())
print(Ratio)

import modules.DWM10_Parms as DWM10_Parms
def hierarchical_document_similarity_with_long_digit(doc1,doc2):
    character_error_tolerance = 2
    digit_tolerance = 6
    S = None
    if doc1 == doc2 or set.issubset(set(doc1), set(doc2)) or set.issubset(set(doc2), set(doc1)):
        S = 1.0
    elif set.isdisjoint(set(doc1), set(doc2)):
        S = 0.0
    else:
        WS = 0
        long_digit_found = False
        for word1 in doc1:
            for word2 in doc2:
                if word1.isalpha() and word2.isalpha() and ((word1 == word2) or (word1 in word2) or (word2 in word1)):
                    WS = WS + 1.0
                elif (word1.isdigit()) and (len(word1) >= digit_tolerance) and (word2.isdigit()) and (len(word2) >= digit_tolerance) and ((word1 == word2) or (word1 in word2) or (word2 in word1)):
                    WS = WS + 1.0
                    long_digit_found = True
                else:
                    L = Levenshtein.ratio(word1,word2)
                    min_length = min([len(word1), len(word2)])
                    max_length = max([len(word1), len(word2)])
                    diff = abs(max_length-min_length)
                    if (L > DWM10_Parms.mu) and (diff>=0 and diff<character_error_tolerance):
                        WS = WS + 1.0
        long_digit_found_weight = 0
        diff = len(doc1) - WS
        if long_digit_found:
            if diff > 0:
                long_digit_found_weight = diff-1
        else:
            long_digit_found_weight = -diff
        S = float((WS+long_digit_found_weight)/(len(doc1)))
        if S > DWM10_Parms.mu:
            S = 1.0
        else:
            S = 0.0
    return S

ref2 = tokenizerCompress("TRACEY AARON LEIGH 176 NOTTIDGE CT CLEMMONS NC 27012 954451341")
# ref2 = tokenizerCompress("MICHAEL AARON SCOT 1010 PALISADE CT MARTINEZ CA 94553 642194591")
# ref1 = tokenizerCompress("MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012")
ref1 = tokenizerCompress("MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012 642-19-4591")
print(hierarchical_document_similarity_with_long_digit(ref1,ref2))
pc1 = set([1,2,3])
pc2 = set([2,3])
print(set.issubset(set(pc2),set(pc1)))




