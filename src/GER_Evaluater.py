
from tqdm import tqdm

def evaluate_clusters(cluster_list, N, ground_truth_file):
    print("Evaluate clusters")
    def countPairs(dict):
        totalPairs = 0
        for cnt in dict.values():
            pairs = cnt*(cnt-1)/2
            totalPairs +=pairs
        return totalPairs
    erDict = {}
    for pair in cluster_list:
        clusterID = pair[0]
        refID = pair[1]
        newPair = (clusterID,'x')
        erDict[refID] = newPair
    truthFile = open(ground_truth_file,'r')
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
    for pair in tqdm(erDict.values()):
        clusterID = pair[0]
        truthID = pair[1]
        if pair in truePos:
            cnt = truePos[pair]
            cnt +=1
            truePos[pair] = cnt
        else:
            truePos[pair] = 1
        if clusterID in linkedPairs:
            cnt = linkedPairs[clusterID]
            cnt +=1
            linkedPairs[clusterID] = cnt
        else:
            linkedPairs[clusterID] = 1
        if truthID in equivPairs:
            cnt = equivPairs[truthID]
            cnt +=1
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
    print('Ground truth pairs (E) = ' + str(E))
    print('True positives (TP) = ' + str(TP))
    print('True negatives (TN) = ' + str(TN))
    print('False positives (FP) = ' + str(FP))
    print('False negatives (FN) = ' + str(FN))
    print('False positive rate (FPR) = ' + str(FPR))
    print('True positive rate (TPR) = ' + str(TPR))
    print('True negative rate (TNR) = ' + str(TNR))
    print('Accuracy = ' + str(accuracy))
    print('Balanced accuracy = ' + str(balanced_accuracy))
    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1-measure = ' + str(F1))
