import math
from collections import defaultdict

import numpy as np
import scipy.sparse


class Tasks():
    # CONSTRUCTOR
    def __init__(self, graph):
        self.graph = graph
        self.m = self.graph.adjacency_weighted.sum() * 0.5
        label_data = []
        label_rows = []
        label_cols = []
        for n, a in self.graph.nodes.items():
            label_rows.append(int(self.graph.node_index[n]))
            label_cols.append(int(a["label"]))
            label_data.append(1)
        max_labels = max(label_cols) + 1
        self.node_label_matrix = scipy.sparse.csc_matrix(
            (np.array(label_data), (np.array(label_rows), np.array(label_cols))), shape=(self.graph.N, max_labels))

    def get_edge_weight(self, n1, n2):
        return float(self.graph.adjacency_weighted[self.graph.node_index[n1], self.graph.node_index[n2]])

    def get_node_label(self, node):
        return self.graph.node_label[node]

    def set_node_label(self, l, node):
        self.graph.node_label[node] = l

    def get_node_real_id(self, node):
        return self.graph.node_real_id[node]

    def get_node_weighted_degree(self, node):
        return float(self.graph.degree_weighted[self.graph.node_index[node], self.graph.node_index[node]])

    def get_node_degree(self, node):
        return float(self.graph.degree[self.graph.node_index[node], self.graph.node_index[node]])

    def get_neighbors(self, node):
        neighbors = []
        neighbor_indices = self.graph.adjacency[:, self.graph.node_index[node]].indices
        for i in neighbor_indices:
            if i == self.graph.node_index[node]:
                continue
            neighbors.append(self.graph.node_inverted_index[i])
        return neighbors

    def get_sum_of_neighborhood_weights(self, node):
        return float(self.graph.adjacency_weighted[:, self.graph.node_index[node]].sum(axis=0)[0][0])

    def get_neighborhood_weights(self, node):
        neighbor_indices = self.graph.adjacency_weighted[:, self.graph.node_index[node]].indices
        weights = []
        for i in neighbor_indices:
            w = self.graph.adjacency_weighted[i, self.graph.node_index[node]]
            weights.append(w)
        return weights

    def get_cluster_members_names(self, cluster_id):
        members = []
        members_indices = list(np.where((self.node_label_matrix.toarray()[:, cluster_id] == 1))[0])
        for i in members_indices:
            members.append(self.graph.node_inverted_index[i])
        # return list({key: value for key, value in self.graph.node_label.items() if value == cluster_id}.keys())
        return members

    def get_cluster_members_indices(self, cluster_id):
        # return list({key: value for key, value in self.graph.node_label.items() if value == cluster_id}.keys())
        return list(np.where((self.node_label_matrix.toarray()[:, cluster_id] == 1))[0])

    def get_sum_of_incident_nodes_to_cluster(self, cluster_id):
        cluster_members = self.get_cluster_members_names(cluster_id)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_incident_nodes_to_cluster_new_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        cluster_members.append(node)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_incident_nodes_to_cluster_without_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        if node in cluster_members:
            cluster_members.remove(node)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_weights_between_node_and_cluster(self, node, j_membership):
        cluster_indices = self.get_cluster_members_indices(j_membership)
        node_index = self.graph.node_index[node]
        return float(self.graph.adjacency_weighted[cluster_indices, node_index].sum())

    def get_sum_of_weights_inside_cluster(self, cluster_id):
        cluster_indices = self.get_cluster_members_indices(cluster_id)
        return float(self.graph.adjacency_weighted[cluster_indices, cluster_indices].sum() * 0.5)

    def get_sum_of_weights_inside_cluster_with_new_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        if node not in cluster_members:
            cluster_members.append(node)
        cluster_indices = []
        for n in cluster_members:
            cluster_indices.append(self.graph.node_index[n])
        return float(self.graph.adjacency_weighted[cluster_indices, cluster_indices].sum() * 0.5)

    def get_sum_of_weights_inside_cluster_without_node(self, cluster_id, node):
        cluster_members = self.get_cluster_members_names(cluster_id)
        if node in cluster_members:
            cluster_members.remove(node)
        cluster_indices = []
        for n in cluster_members:
            cluster_indices.append(self.graph.node_index[n])
        return float(self.graph.adjacency_weighted[cluster_indices, cluster_indices].sum() * 0.5)

    # MODULARITY FUNCTIONS
    def compute_modularity(self):
        res = 0
        for com, nodes in self.graph.node_label_profile.items():
            for i in nodes:
                for j in nodes:
                    a_i_j = self.get_edge_weight(i, j)
                    k_i = self.get_sum_of_neighborhood_weights(i)
                    k_j = self.get_sum_of_neighborhood_weights(j)
                    res = res + (a_i_j - ((k_i * k_j) / (2 * self.m)))
        modularity = res / (2 * self.m)
        return modularity

    def compute_delta_modularity_node_add(self, cluster, node):
        sigma_in = self.get_sum_of_weights_inside_cluster_with_new_node(cluster, node)
        sigma_total = self.get_sum_of_incident_nodes_to_cluster_new_node(cluster, node)
        k_i_in = self.get_sum_of_weights_between_node_and_cluster(node, cluster)
        k_i = self.get_sum_of_neighborhood_weights(node)
        delta_modularity = (((sigma_in + k_i_in) / (2 * self.m)) - ((sigma_total + k_i) / (2 * self.m)) ** 2) - (
                (sigma_in / (2 * self.m)) - ((sigma_total / (2 * self.m)) ** 2) - ((k_i / (2 * self.m)) ** 2))
        return delta_modularity

    def compute_delta_modularity_node_remove(self, cluster, node):
        sigma_in = self.get_sum_of_weights_inside_cluster_without_node(cluster, node)
        sigma_total = self.get_sum_of_incident_nodes_to_cluster_without_node(cluster, node)
        k_i_in = self.get_sum_of_weights_between_node_and_cluster(node, cluster)
        k_i = self.get_sum_of_neighborhood_weights(node)
        delta_modularity = (((sigma_in + k_i_in) / (2 * self.m)) - ((sigma_total + k_i) / (2 * self.m)) ** 2) - (
                (sigma_in / (2 * self.m)) - ((sigma_total / (2 * self.m)) ** 2) - ((k_i / (2 * self.m)) ** 2))
        return delta_modularity

    # ATTRIBUTE TASKS
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

    def compute_graph_entropy(self):
        communities_grouped = defaultdict(list)
        for key, value in sorted(self.graph.node_label_profile.items()):
            communities_grouped[value].append(key)
        communities_grouped = dict(communities_grouped)
        entropies = []
        for cluster, records in communities_grouped.items():
            cluster = []
            for r in records:
                cluster.append(self.graph.node_attribute[r])
            quality = self.__calculate_cluster_entropy(cluster)
            entropies.append(quality)
        return round(np.mean(np.array(entropies)), 4)
