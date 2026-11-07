import numpy as np


class Tasks():
    # CONSTRUCTOR
    def __init__(self, graph):
        self.graph = graph
        self.m = self.graph.adjacency_weighted.sum() * 0.5

    def get_edge_weight(self, n1, n2):
        return float(self.graph.adjacency_weighted[self.graph.node_index[n1], self.graph.node_index[n2]])

    def get_node_label(self, node):
        return self.graph.node_label[node]

    def set_node_label(self, l, node):
        self.graph.node_label[node] = l

    def get_neighbors(self, node):
        index_of_node = self.graph.node_index[node]
        node_column = self.graph.adjacency.toarray()[:][index_of_node]
        neighbors = []
        neighbor_indices = np.where(node_column != 0)[0]
        for i in neighbor_indices:
            neighbors.append(self.graph.node_inverted_index[i])
        return neighbors

    def get_sum_of_neighborhood_weights(self, node):
        return float(self.graph.adjacency_weighted[:, self.graph.node_index[node]].sum(axis=0)[0][0])

    def get_sum_of_incident_nodes_to_cluster(self, cluster_id):
        cluster_members = self.get_cluster_members(cluster_id)
        sum = 0
        for n in cluster_members:
            neighbors = self.get_neighbors(n)
            for nbr in neighbors:
                if nbr not in cluster_members:
                    sum = sum + self.get_edge_weight(n, nbr)
        return sum

    def get_sum_of_weights_between_node_and_cluster(self, node, j_membership):
        cluster_members = self.get_cluster_members(j_membership)
        cluster_indices = []
        for n in cluster_members:
            cluster_indices.append(self.graph.node_index[n])
        node_index = self.graph.node_index[node]
        return float(self.graph.adjacency_weighted[cluster_indices, node_index].sum())

    def get_cluster_members(self, cluster_id):
        return list({key: value for key, value in self.graph.node_label.items() if value == cluster_id}.keys())

    def get_sum_of_weights_inside_cluster(self, cluster_id):
        cluster_members = self.get_cluster_members(cluster_id)
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

    def compute_delta_modularity(self, cluster, node):
        sigma_in = self.get_sum_of_weights_inside_cluster(cluster)
        sigma_total = self.get_sum_of_incident_nodes_to_cluster(cluster)
        k_i_in = self.get_sum_of_weights_between_node_and_cluster(node, cluster)
        k_i = self.get_sum_of_neighborhood_weights(node)
        delta_modularity = (((sigma_in + k_i_in) / (2 * self.m)) - ((sigma_total + k_i) / (2 * self.m)) ** 2) - (
                (sigma_in / (2 * self.m)) - ((sigma_total / (2 * self.m)) ** 2) - ((k_i / (2 * self.m)) ** 2))
        return delta_modularity
