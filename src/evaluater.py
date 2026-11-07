from importer import *


class EvaluatER():

    def __init__(self, clusters, truth_file):
        self.clusters = clusters
        self.truth_file = truth_file
        self.n = 0
        for cc in clusters:
            for r in cc:
                self.n = self.n + 1
        self.modularity = 0
        self.quality = 0
        self.profile = {}
        self.L, self.E, self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0, 0, 0
        self.FPR, self.TPR, self.TNR, self.accuracy, self.balanced_accuracy, self.precision = 0, 0, 0, 0, 0, 0
        self.recall, self.F1 = 0, 0
        self.link_index_pairs = []
        self.link_index_dict = {}
        self.__convert_node_labels_to_link_index()

    def __convert_node_labels_to_link_index(self):
        for c in self.clusters:
            minimum_node = min(c)
            for n in c:
                self.link_index_pairs.append((minimum_node, n))
                self.link_index_dict[n] = minimum_node

    @staticmethod
    def evaluate_samples(self):
        y_true = []
        y_pred = []
        i = 0
        for n, l in self.graph_object.node_labels.items():
            predicted_label = str(np.argmax(self.predictions[i]) + 1)
            actual_label = str(l)
            y_pred.append(predicted_label)
            y_true.append(actual_label)
            i = i + 1
        labels = list(set(list(self.graph_object.node_labels.values())))
        cm = skl.metrics.confusion_matrix(y_true, y_pred, labels=labels)

        diagonal = np.diag(cm)
        mean_true_positives = np.mean(diagonal)

        tns = []

        for i, v in enumerate(diagonal):
            indices = list(diagonal)
            del indices[i]
            sum_of_indices = np.sum(np.array(indices))
            tns.append(sum_of_indices)
        mean_true_negatives = np.mean(np.array(tns))

        fps = []
        fns = []
        for index1, row1 in enumerate(cm):
            class_specific_false_positives = 0
            class_specific_false_negatives = 0
            items = row1
            for index2, row2 in enumerate(cm):
                if index1 != index2:
                    left_side = items[0:index1]
                    right_side = items[index1 + 1:]
                    class_specific_false_positives = class_specific_false_positives + sum(left_side)
                    class_specific_false_negatives = class_specific_false_negatives + sum(right_side)
            fps.append(class_specific_false_positives)
            fns.append(class_specific_false_negatives)
        mean_false_positives = np.mean(np.array(fps))
        mean_false_negatives = np.mean(np.array(fns))

        precision = mean_true_positives / (mean_true_positives + mean_false_positives)
        recall = mean_true_positives / (mean_false_negatives + mean_true_positives)
        accuracy = (mean_true_positives + mean_true_negatives) / self.input_dim
        f1_score = (2 * precision * recall) / (precision + recall)

        return precision, recall, accuracy, f1_score

    @staticmethod
    def evaluate_embeddings(self):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for i, ne1 in enumerate(self.embeddings):
            distances = []
            source_node = self.graph_object.node_index[i]
            for j, ne2 in enumerate(self.embeddings):
                target_node = self.graph_object.node_index[j]
                if source_node == target_node:
                    continue
                else:
                    match_cosine_distance = sp.spatial.distance.cosine(np.array(ne1), np.array(ne2))
                    if match_cosine_distance < self.cosine_threshold:
                        distances.append(match_cosine_distance)
            if len(distances) == 0:
                distances.append([1])
            distances = np.array(distances)
            least_distance_index = np.argmin(distances)
            source_node_actual_label = self.graph_object.node_labels[i]
            source_node_predicted_label = np.argmax(self.predictions[i]) + 1
            target_node_actual_label = self.graph_object.node_labels[least_distance_index]
            target_node_predicted_label = np.argmax(self.predictions[least_distance_index]) + 1
            if source_node_actual_label == target_node_actual_label and source_node_predicted_label == target_node_predicted_label:
                true_positives = true_positives + 1
            elif source_node_actual_label == target_node_actual_label and source_node_predicted_label != target_node_predicted_label:
                false_negatives = false_negatives + 1
            elif source_node_actual_label != target_node_actual_label and source_node_predicted_label == target_node_predicted_label:
                false_positives = false_positives + 1
            else:
                true_negatives = true_negatives + 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        accuracy = (true_positives + true_negatives) / self.input_dim
        f1_score = (2 * precision * recall) / (precision + recall)

        return precision, recall, accuracy, f1_score

    @staticmethod
    def evaluate_clusters(self):
        y_true = []
        y_pred = []
        i = 0
        for n, l in self.graph_object.node_labels.items():
            predicted_label = str(np.argmax(self.predictions[i]) + 1)
            actual_label = str(l)
            y_pred.append(predicted_label)
            y_true.append(actual_label)
            i = i + 1
        return skl.metrics.cluster.adjusted_rand_score(y_true, y_pred)

    @staticmethod
    def __count_pairs(dict):
        totalPairs = 0
        for cnt in dict.values():
            pairs = cnt * (cnt - 1) / 2
            totalPairs += pairs
        return totalPairs

    def __generate_profile(self):
        profileDict = {}
        clusterSizeDict = {}
        for key in self.link_index_dict:
            clusterKey = self.link_index_dict[key]
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

    def __generate_metrics(self):
        erDict = {}
        for refID in self.link_index_dict:
            clusterID = self.link_index_dict[refID]
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
        self.TN = abs(float((self.n * (self.n - 1)) / 2) - (
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
        self.__generate_profile()
        print("---------------------------")
        print("Linked pairs (L): " + str(self.L))
        print("---------------------------")
        print("Ground truth pairs (E): " + str(self.E))
        print("---------------------------")
        print("True positives (TP): " + str(self.TP))
        print("---------------------------")
        print("True negatives (TN): " + str(self.TN))
        print("---------------------------")
        print("False positives (FP): " + str(self.FP))
        print("---------------------------")
        print("False negatives (FN): " + str(self.FN))
        print("---------------------------")
        print("False positive rate (FPR): " + str(self.FPR))
        print("---------------------------")
        print("True positive rate (TPR): " + str(self.TPR))
        print("---------------------------")
        print("True negative rate (TNR): " + str(self.TNR))
        print("---------------------------")
        print("Accuracy: " + str(self.accuracy))
        print("---------------------------")
        print("Balanced accuracy: " + str(self.balanced_accuracy))
        print("---------------------------")
        print("Precision: " + str(self.precision))
        print("---------------------------")
        print("Recall: " + str(self.recall))
        print("---------------------------")
        print("F1-score: " + str(self.F1))
        print("---------------------------")
        print("\nCluster profile")
        print("Size\tcount")
        total = 0
        for key in sorted(self.profile.keys()):
            clusterTotal = key * self.profile[key]
            total += clusterTotal
            print(key, "\t", self.profile[key], "\t", clusterTotal)
        print("\tTotal\t", total)
        print("---------------------------")
