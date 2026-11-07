import re

import Levenshtein

import modules.DWM10_Parms as DWM10_Parms


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



def removeStopWords(tokenList, tokenFreqDict):
    newList = []
    # print('-- tokenList', tokenList)
    for token in tokenList:
        tokenLen = len(token)
        includeToken = True
        freq = tokenFreqDict[token]
        if freq >= DWM10_Parms.sigma:
            includeToken = False
            # print('-- sigma rule', token, freq)
        if DWM10_Parms.removeExcludedBlkTokens:
            if tokenLen < DWM10_Parms.minBlkTokenLen:
                includeToken = False
                # print('-- min len rule', token, tokenLen)
            if token.isdigit() and DWM10_Parms.excludeNumericBlocks:
                includeToken = False
                # print('-- number rule', token)
        if DWM10_Parms.removeDuplicateTokens and (token in newList):
            includeToken = False
            # print('-- duplicate token rule', token)
        if includeToken:
            newList.append(token)
    # print('-- newList', newList)
    return newList


def decision(doc1, doc2, tokenize, remove_stop_words, token_frequencies, digit_tolerance=6, typo_threshold = 2):
    if tokenize:
        doc1 = tokenizerCompress(doc1)
        doc2 = tokenizerCompress(doc2)
    if remove_stop_words:
        doc1 = removeStopWords(doc1, token_frequencies)
        doc2 = removeStopWords(doc2, token_frequencies)
    S = None
    if doc1 == doc2 or set.issubset(set(doc1), set(doc2)) or set.issubset(set(doc2), set(doc1)):
        S = 1.0
    elif set.isdisjoint(set(doc1), set(doc2)):
        S = 0.0
    else:
        WS = 0
        # long_digit_found = False
        token_equivalence_condition = lambda word1, word2: (word1 == word2) or (word1 in word2) or (word2 in word1) or (Levenshtein.distance(word1,word2)<=typo_threshold)
        numerical_token_discriminatory_condition =  lambda word1, word2: word1.isdigit() and word2.isdigit() and (len(word1) >= digit_tolerance) and (len(word2) >= digit_tolerance) and token_equivalence_condition(word1,word2)
        for word1 in doc1:
            for word2 in doc2:
                if token_equivalence_condition(word1,word2):
                    WS = WS + 1.0
                elif numerical_token_discriminatory_condition(word1,word2):
                    WS = WS + 1.0
                    # long_digit_found = True
        # long_digit_found_weight = 0
        # diff = len(doc1) - WS
        # if long_digit_found:
        #     if diff > 0:
        #         long_digit_found_weight = diff - 1
        # else:
        #     long_digit_found_weight = -diff
        # S = float((WS + long_digit_found_weight) / (len(doc1)))
        S = float(WS / max(len(doc1),len(doc2)))
        # if S > 0:
        #     S = 1.0
        # else:
        #     S = 0.0
    return S


# A735485,CHRISTOPHER,ABBOTT,LEE,137 COLUMBINE DR # 502,WINSTON SALEM,NC,27106,173-61-6732
# A782781,CHRISTOPHER,ABBOTT,SR,2300 MCKINNON CT # G,CLEMMONS,NC,27012,184-85-8365

# # ref2 = "TRACEY AARON LEIGH 176 NOTTIDGE CT CLEMMONS NC 27012 954451341"
# ref2 = "MICHAEL AARON SCOT 1010 PALISADE CT MARTINEZ CA 94553 642194591"
# # ref1 = tokenizerCompress("MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012")
# ref1 = "MICHAEL AARON SCOT 176 NOTTIDGE CT CLEMMONS NC 27012 642-19-4591"



# ref1 = "CHRISTOPHER ABBOTT LEE 137 COLUMBINE DR 502 WINSTON SALEM NC 27106 173616732"
ref1 = "CHRISTOPHER ABBOTT LEE 137 COLUMBINE DR 502 WINSTON SALEM NC 27106 173616732"
ref2 = "CHRISTOPHER ABBOTT SR 2300 MCKINNON CT G CLEMMONS NC 27012 184858365"
token_frequencies = {}
print(decision(ref1, ref2, True, False, token_frequencies))
