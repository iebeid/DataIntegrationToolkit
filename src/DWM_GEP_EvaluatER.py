import math
from collections import defaultdict

import networkx as nx
import numpy as np

from modules import DWM10_Parms


class Evaluater():

    def __init__(self, graph, truth_file):
        self.graph = graph
        self.truth_file = truth_file
        self.N = len(self.graph.nodes)
        self.modularity = 0
        self.quality = 0
        self.profile = {}
        self.L, self.E, self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0, 0, 0
        self.FPR, self.TPR, self.TNR, self.accuracy, self.balanced_accuracy, self.precision = 0, 0, 0, 0, 0, 0
        self.recall, self.F1 = 0, 0

    def print_clusters(self):
        res = defaultdict(list)
        for key, val in sorted(self.graph.node_label.items()):
            res[val].append(key)
        print("Number of entity clusters: " + str(len(res.keys())))
        sum = 0
        for c in res.values():
            sum = sum + len(c)
            print(c)
            print(len(c))
        print("Number of nodes clustered: " + str(sum))

    @staticmethod
    def __count_pairs(dict):
        totalPairs = 0
        for cnt in dict.values():
            pairs = cnt * (cnt - 1) / 2
            totalPairs += pairs
        return totalPairs

    def __calculate_cluster_entropy(self, cluster):
        tokenCount = 0
        refCnt = len(cluster)
        if refCnt > 1:
            for j in range(0, refCnt):
                tokenCount = tokenCount + len(cluster[j])
            baseProb = 1 / float(refCnt)
            base = -tokenCount * baseProb * math.log(baseProb, 2)
            entropy = 0.0
            clusterSize = len(cluster)
            for j in range(0, len(cluster) - 1):
                jList = cluster[j]
                for token in jList:
                    cnt = 1
                    for k in range(j + 1, len(cluster)):
                        if token in cluster[k]:
                            cnt += 1
                            cluster[k].remove(token)
                    tokenProb = cnt / clusterSize
                    term = -tokenProb * math.log(tokenProb, 2)
                    entropy += term
                    quality = 1.0 - entropy / base
                    cnt = 0
            for token in cluster[clusterSize - 1]:
                tokenProb = 1.0 / clusterSize
                term = -tokenProb * math.log(tokenProb, 2)
                entropy += term
                quality = 1.0 - entropy / base
            quality = 1.0 - entropy / base
        else:
            quality = 1.0
        return float(quality)

    def __generate_profile(self):
        profileDict = {}
        clusterSizeDict = {}
        for key in self.graph.link_index_dict:
            clusterKey = self.graph.link_index_dict[key]
            if clusterKey not in clusterSizeDict:
                clusterSizeDict[clusterKey] = 1
            else:
                cnt = clusterSizeDict[clusterKey]
                cnt += 1
                clusterSizeDict[clusterKey] = cnt
        for key in clusterSizeDict:
            clusterSize = clusterSizeDict[key]
            if clusterSize not in profileDict:
                profileDict[clusterSize] = 1
            else:
                cnt = profileDict[clusterSize]
                cnt += 1
                profileDict[clusterSize] = cnt
        self.profile = profileDict

    def __compute_final_modularity(self):
        com = 0
        final_communities = {}
        communities_grouped = defaultdict(list)
        for key, value in sorted(self.graph.link_index_dict.items()):
            communities_grouped[value].append(key)
        communities_grouped = dict(communities_grouped)
        for cluster, nodes in communities_grouped.items():
            for n in nodes:
                final_communities[n] = com
            com = com + 1
        self.modularity = round(self.graph.tasks.compute_modularity(), 4)

    def __compute_final_entropy(self):
        communities_grouped = defaultdict(list)
        for key, value in sorted(self.graph.link_index_dict.items()):
            communities_grouped[value].append(key)
        communities_grouped = dict(communities_grouped)
        entropies = []
        for cluster, records in communities_grouped.items():
            cluster = []
            for r in records:
                cluster.append(self.graph.node_attribute[r])
            quality = self.__calculate_cluster_entropy(cluster)
            entropies.append(quality)
        self.quality = round(np.mean(np.array(entropies)), 4)

    def __generate_metrics(self):

        erDict = {}
        for refID in self.graph.link_index_dict:
            clusterID = self.graph.link_index_dict[refID]
            erDict[refID] = (clusterID, 'x')
        truthFile = open(self.truth_file, 'r')
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
        self.L = self.__count_pairs(linkedPairs)
        self.E = self.__count_pairs(equivPairs)
        self.TP = self.__count_pairs(truePos)
        self.FP = float(self.L - self.TP)
        self.FN = float(self.E - self.TP)
        self.TN = abs(float((self.N * (self.N - 1)) / 2) - (
                self.TP + self.FP + self.FN))  # Pairs that were not linked and should not have been
        if self.L > 0:
            self.precision = self.TP / float(self.L)
        else:
            self.precision = 1.00
        if self.E > 0:
            self.recall = self.TP / float(self.E)
        else:
            self.recall = 1.00
        if self.precision != 0 and self.recall != 0:
            self.F1 = round((2 * self.precision * self.recall) / (self.precision + self.recall), 4)
            self.FPR = round((self.FP / (self.FP + self.TN)), 4)
            self.TPR = round((self.TP / (self.TP + self.FN)), 4)
            self.TNR = round((1 - self.FPR), 4)
            self.accuracy = round(((self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)), 4)
            self.balanced_accuracy = round(((self.TPR + self.TNR) / 2), 4)
            self.precision = round(self.precision, 4)
            self.recall = round(self.recall, 4)

    def run(self):
        self.__generate_metrics()
        self.__compute_final_modularity()
        self.__compute_final_entropy()
        self.__generate_profile()
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Linked pairs (L): " + str(self.L))
        print("Linked pairs (L): " + str(self.L), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Ground truth pairs (E): " + str(self.E))
        print("Ground truth pairs (E): " + str(self.E), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("True positives (TP): " + str(self.TP))
        print("True positives (TP): " + str(self.TP), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("True negatives (TN): " + str(self.TN))
        print("True negatives (TN): " + str(self.TN), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("False positives (FP): " + str(self.FP))
        print("False positives (FP): " + str(self.FP), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("False negatives (FN): " + str(self.FN))
        print("False negatives (FN): " + str(self.FN), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("False positive rate (FPR): " + str(self.FPR))
        print("False positive rate (FPR): " + str(self.FPR), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("True positive rate (TPR): " + str(self.TPR))
        print("True positive rate (TPR): " + str(self.TPR), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("True negative rate (TNR): " + str(self.TNR))
        print("True negative rate (TNR): " + str(self.TNR), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Accuracy: " + str(self.accuracy))
        print("Accuracy: " + str(self.accuracy), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Balanced accuracy: " + str(self.balanced_accuracy))
        print("Balanced accuracy: " + str(self.balanced_accuracy), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Precision: " + str(self.precision))
        print("Precision: " + str(self.precision), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Recall: " + str(self.recall))
        print("Recall: " + str(self.recall), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("F1-score: " + str(self.F1))
        print("F1-score: " + str(self.F1), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Final modularity: " + str(self.modularity))
        print("Final modularity: " + str(self.modularity), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("Average cluster quality: " + str(self.quality))
        print("Average cluster quality: " + str(self.quality), file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
        print("\nCluster profile")
        print("\nCluster profile", file=DWM10_Parms.logFile)
        print("Size\tcount")
        print("Size\tcount", file=DWM10_Parms.logFile)
        total = 0
        for key in sorted(self.profile.keys()):
            clusterTotal = key * self.profile[key]
            total += clusterTotal
            print(key, "\t", self.profile[key], "\t", clusterTotal)
            print(key, "\t", self.profile[key], "\t", clusterTotal, file=DWM10_Parms.logFile)
        print("\tTotal\t", total)
        print("\tTotal\t", total, file=DWM10_Parms.logFile)
        print("---------------------------")
        print("---------------------------", file=DWM10_Parms.logFile)
