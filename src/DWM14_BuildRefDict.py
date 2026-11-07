#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import sys
import chardet
import codecs
import modules.DWM10_Parms as DWM10_Parms


def tokenizeInput():
    # ***********Inner Function*******************************
    # Replace delimiter with blanks, then compress token by replacing non-word characters with null
    def tokenizerCompress(string):
        string = string.upper()
        string = string.replace(delimiter, ' ')
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

    # ***********Outer Main Function*******************************
    # Start of Main Tokenizer Function
    logFile = DWM10_Parms.logFile
    print('\n>> Starting DWM14')
    print('\n>> Starting DWM14', file=logFile)
    inputFileName = DWM10_Parms.inputFileName
    refDict = {}
    print('Input Reference File Name =', inputFileName)
    print('Input Reference File Name =', inputFileName, file=logFile)
    hasHeader = DWM10_Parms.hasHeader
    print('Input File has Header Records =', hasHeader)
    print('Input File has Header Records =', hasHeader, file=logFile)
    delimiter = DWM10_Parms.delimiter
    print('Input File Delimiter =', delimiter)
    print('Input File Delimiter =', delimiter, file=logFile)
    tokenizerType = DWM10_Parms.tokenizerType
    print('Tokenizer Function Type =', tokenizerType)
    print('Tokenizer Function Type =', tokenizerType, file=logFile)
    removeDuplicateTokens = DWM10_Parms.removeDuplicateTokens
    print('Remove Duplicate Reference Tokens =', removeDuplicateTokens)
    print('Remove Duplicate Reference Tokens =', removeDuplicateTokens, file=logFile)
    goodType = False
    if tokenizerType == 'Splitter':
        tokenizerFunction = tokenizerSplitter
        goodType = True
    if tokenizerType == 'Compress':
        tokenizerFunction = tokenizerCompress
        goodType = True
    if goodType == False:
        print('**Error: Invalid Parameter value for tokenizerType ', tokenizerType)
        sys.exit()
    # Read input file and build reference dictionary (refDict)
    encoding = None
    with open(inputFileName, "rb") as rawdata:
        result = chardet.detect(rawdata.read())
        encoding = result["encoding"]
        if encoding == "ascii":
            encoding="utf-8"
    inputFile = open(inputFileName, 'r', encoding=encoding)
    refCnt = 0
    # skip header record
    if hasHeader:
        line = inputFile.readline()
    line = inputFile.readline()
    tokenCnt = 0
    tokensOut = 0
    tokenFreqCount = []
    while line != '':
        refCnt += 1
        line = line.strip()
        # assume first token is the reference identifier (refID)
        firstDelimiter = line.find(delimiter)
        refID = line[0:firstDelimiter]
        body = line[firstDelimiter + 1:]
        tokenList = tokenizerFunction(body)
        tokenCnt = tokenCnt + len(tokenList)
        if removeDuplicateTokens:
            tokenList = list(dict.fromkeys(tokenList))
        tokensOut = tokensOut + len(tokenList)
        refDict[refID] = tokenList
        line = inputFile.readline()
    inputFile.close()
    print('Total References Read=', refCnt)
    print('Total References Read=', refCnt, file=logFile)
    print('Total Tokens Found =', tokenCnt)
    print('Total Tokens Found =', tokenCnt, file=logFile)
    return refDict
