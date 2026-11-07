#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np

import modules.DWM10_Parms as DWM10_Parms


def generateMetricsFull(logFile, N, cluster_modularities, linkIndex, truthFileName):
    def countPairs(dict):
        totalPairs = 0
        for cnt in dict.values():
            pairs = cnt * (cnt - 1) / 2
            totalPairs += pairs
        return totalPairs

    print('\n>>Starting DWM99\n')
    print('\n>>Starting DWM99\n', file=logFile)
    print('Truth File Name = ', truthFileName)
    print('Truth File Name = ', truthFileName, file=logFile)
    erDict = {}
    for pair in linkIndex:
        clusterID = pair[0]
        if clusterID != 'X':
            refID = pair[1]
            newPair = (clusterID, 'x')
            erDict[refID] = newPair
    truthFile = open(truthFileName, 'r')
    line = (truthFile.readline()).strip()
    while line != '':
        part = line.split(',')
        recID = part[0].strip()
        truthID = part[1].strip()
        if recID in erDict:
            oldPair = erDict[recID]
            clusterID = oldPair[0]
            newPair = (clusterID, truthID)
            erDict[recID] = newPair
        line = (truthFile.readline()).strip()
    linkedPairs = {}
    equivPairs = {}
    truePos = {}
    for pair in erDict.values():
        clusterID = pair[0]
        truthID = pair[1]
        if pair in truePos:
            cnt = truePos[pair]
            cnt += 1
            truePos[pair] = cnt
        else:
            truePos[pair] = 1
        if clusterID in linkedPairs:
            cnt = linkedPairs[clusterID]
            cnt += 1
            linkedPairs[clusterID] = cnt
        else:
            linkedPairs[clusterID] = 1
        if truthID in equivPairs:
            cnt = equivPairs[truthID]
            cnt += 1
            equivPairs[truthID] = cnt
        else:
            equivPairs[truthID] = 1
    # End of counts
    # Compute measures
    L = countPairs(linkedPairs)  # Pairs that were linked
    E = countPairs(equivPairs)  # Pairs ground truth
    TP = countPairs(truePos)  # Pairs that were linked correctly
    FP = float(L - TP)  # Pairs that were linked but shouldn't have been
    FN = float(E - TP)  # Pairs that were not linked but should have been
    TN = abs(float((N * (N - 1)) / 2) - (TP + FP + FN))  # Pairs that were not linked and should not have been
    if L > 0:
        precision = round(TP / float(L), 4)
    else:
        precision = 1.00
    if E > 0:
        recall = round(TP / float(E), 4)
    else:
        recall = 1.00
    F1 = round((2 * precision * recall) / (precision + recall), 4)
    FPR = round((FP / (FP + TN)), 4)
    TPR = round((TP / (TP + FN)), 4)
    TNR = round((1 - FPR), 4)
    accuracy = round(((TP + TN) / (TP + TN + FP + FN)), 4)
    balanced_accuracy = round(((TPR + TNR) / 2), 4)
    print('Linked pairs (L) = ' + str(L))
    print('Linked pairs (L) = ' + str(L), file=logFile)
    print('Ground truth pairs (E) = ' + str(E))
    print('Ground truth pairs (E) = ' + str(E), file=logFile)
    print('True positives (TP) = ' + str(TP))
    print('True positives (TP) = ' + str(TP), file=logFile)
    print('True negatives (TN) = ' + str(TN))
    print('True negatives (TN) = ' + str(TN), file=logFile)
    print('False positives (FP) = ' + str(FP))
    print('False positives (FP) = ' + str(FP), file=logFile)
    print('False negatives (FN) = ' + str(FN))
    print('False negatives (FN) = ' + str(FN), file=logFile)

    print('False positive rate (FPR) = ' + str(FPR))
    print('False positive rate (FPR) = ' + str(FPR), file=logFile)
    print('True positive rate (TPR) = ' + str(TPR))
    print('True positive rate (TPR) = ' + str(TPR), file=logFile)
    print('True negative rate (TNR) = ' + str(TNR))
    print('True negative rate (TNR) = ' + str(TNR), file=logFile)
    print('Accuracy = ' + str(accuracy))
    print('Accuracy = ' + str(accuracy), file=logFile)
    print('Balanced accuracy = ' + str(balanced_accuracy))
    print('Balanced accuracy = ' + str(balanced_accuracy), file=logFile)

    print('Precision = ' + str(precision))
    print('Precision = ' + str(precision), file=logFile)
    print('Recall = ' + str(recall))
    print('Recall = ' + str(recall), file=logFile)
    print('F1-measure = ' + str(F1))
    print('F1-measure = ' + str(F1), file=logFile)
    initial_modularities = []
    final_modularities = []
    for k, v in cluster_modularities.items():
        initial_modularities.append(float(v["initial"]))
        final_modularities.append(float(v["final"]))
    average_initial_modularity = np.mean(np.array(initial_modularities))
    average_final_modularity = np.mean(np.array(final_modularities))

    print('Average initial modularity = ' + str(average_initial_modularity))
    print('Average initial modularity = ' + str(average_initial_modularity), file=logFile)
    print('Average final modularity = ' + str(average_final_modularity))
    print('Average final modularity = ' + str(average_final_modularity), file=logFile)


def generateMetricsBasic(N, linkIndex):
    def countPairs(dict):
        totalPairs = 0
        for cnt in dict.values():
            pairs = cnt * (cnt - 1) / 2
            totalPairs += pairs
        return totalPairs

    logFile = DWM10_Parms.logFile
    print('\n>>Starting DWM99')
    print('>>Starting DWM99', file=logFile)
    truthFileName = DWM10_Parms.truthFileName
    print('Truth File Name=', truthFileName)
    print('Truth File Name=', truthFileName, file=logFile)

    erDict = {}
    for refID in linkIndex:
        clusterID = linkIndex[refID]
        erDict[refID] = (clusterID, 'x')
    truthFile = open(truthFileName, 'r')
    line = (truthFile.readline()).strip()
    line = (truthFile.readline()).strip()
    while line != '':
        part = line.split(',')
        recID = part[0].strip()
        truthID = part[1].strip()
        if recID in erDict:
            oldPair = erDict[recID]
            clusterID = oldPair[0]
            newPair = (clusterID, truthID)
            erDict[recID] = newPair
        line = (truthFile.readline()).strip()
    linkedPairs = {}
    equivPairs = {}
    truePos = {}
    clusterIndex = []
    for pair in erDict.values():
        clusterID = pair[0]
        truthID = pair[1]
        if pair in truePos:
            cnt = truePos[pair]
            aPair = [pair[0], truthID]
            clusterIndex.append(aPair)
            cnt += 1
            truePos[pair] = cnt
        else:
            truePos[pair] = 1
        if clusterID in linkedPairs:
            cnt = linkedPairs[clusterID]
            cnt += 1
            linkedPairs[clusterID] = cnt
        else:
            linkedPairs[clusterID] = 1
        if truthID in equivPairs:
            cnt = equivPairs[truthID]
            cnt += 1
            equivPairs[truthID] = cnt
        else:
            equivPairs[truthID] = 1
            # End of counts
    L = countPairs(linkedPairs)
    E = countPairs(equivPairs)
    TP = countPairs(truePos)
    FP = float(L - TP)
    FN = float(E - TP)
    TN = abs(float((N * (N - 1)) / 2) - (TP + FP + FN))  # Pairs that were not linked and should not have been
    if L > 0:
        precision = round(TP / float(L), 4)
    else:
        precision = 1.00
    if E > 0:
        recall = round(TP / float(E), 4)
    else:
        recall = 1.00

    fmeas = round((2 * precision * recall) / (precision + recall), 4)

    # for report process
    DWM10_Parms.precision = precision
    DWM10_Parms.recall = recall
    DWM10_Parms.fmeasure = fmeas
    DWM10_Parms.truePairs = TP
    DWM10_Parms.expectedPairs = E
    DWM10_Parms.linkedPairs = L

    print('True Pairs =', TP)
    print('True Pairs =', TP, file=logFile)
    print('Expected Pairs =', E)
    print('Expected Pairs =', E, file=logFile)
    print('Linked Pairs =', L)
    print('Linked Pairs =', L, file=logFile)
    print('Precision=', precision)
    print('Precision=', precision, file=logFile)
    print('Recall=', recall)
    print('Recall=', recall, file=logFile)
    print('F-measure=', fmeas)
    print('F-measure=', fmeas, file=logFile)
