from collections import defaultdict
import numpy as np
import networkx as nx


class Graph:
    def __init__(self, nodes, edges):
        self.edges = edges
        self.node_index = {}
        if nodes:
            self.nodes = nodes
        else:
            self.nodes = {}
            self.index_nodes()
        self.index_edges()
        self.N = len(self.nodes.keys())
        self.number_of_edges = len(self.edges)
        self.node_labels = {}
        self.node_label_profile = {}
        self.create_node_labels()
        self.number_of_classes = int(len(self.node_label_profile.keys()))
        self.y = []

    def index_nodes(self):
        all_nodes = []
        for pair in self.edges:
            source_node = str(pair[0])
            target_node = str(pair[1])
            all_nodes.append(source_node)
            all_nodes.append(target_node)
        all_nodes = list(np.unique(np.array(all_nodes)))
        for i in range(len(all_nodes)):
            self.nodes[all_nodes[i]] = int(i)
            self.node_index[int(i)] = all_nodes[i]

    def index_edges(self):
        indexed_edge_list = []
        for es in self.edges:
            indexed_edge_list.append((self.nodes[es[0]], self.nodes[es[1]], es[2]))
        self.edges = indexed_edge_list

    def create_node_labels(self):
        weightless_edges = []
        for e in self.edges:
            weightless_edges.append((e[0], e[1]))
        data = nx.Graph(weightless_edges)
        connected_components = nx.connected_components(data)
        if len(list(connected_components)) > 1:
            print("WARNING: Graph has more than one component")
        louvain_communities = nx.community.louvain_communities(data, seed=123)
        if len(list(louvain_communities)) < 2:
            print("WARNING: Graph has only one Louvain community")
        c = 0
        for community in louvain_communities:
            c = c + 1
            for n in community:
                self.node_labels[n] = str(c)
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_labels.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)

    def balanced_node_type_sampler(self, split_percent):
        nodes = list(self.nodes.values())
        k = np.floor((split_percent * len(nodes)) / 100)
        number_of_classes = len(list(self.node_label_profile.keys()))
        sample_label_size = np.floor(k / number_of_classes)
        final_samples = []
        for labels, ns in self.node_label_profile.items():
            number_of_samples = 0
            if sample_label_size > len(ns):
                number_of_samples = len(ns)
            elif sample_label_size <= len(ns):
                number_of_samples = sample_label_size
            samples = list(np.random.choice(ns, size=int(number_of_samples), replace=False))
            final_samples = final_samples + samples
        train_samples = list(set(final_samples))
        rest_samples = list(set(nodes) - set(train_samples))
        middle_index = np.math.floor(len(rest_samples) / 2)
        valid_samples = rest_samples[0:middle_index]
        test_samples = rest_samples[middle_index:]
        train_samples_indices = []
        for t_s in train_samples:
            train_samples_indices.append(t_s)
        valid_samples_indices = []
        for t_s in valid_samples:
            valid_samples_indices.append(t_s)
        test_samples_indices = []
        for t_s in test_samples:
            test_samples_indices.append(t_s)

        train_mask = np.zeros((len(nodes)), dtype=int)
        train_mask[train_samples_indices] = 1
        train_mask = train_mask.astype(bool)
        valid_mask = np.zeros((len(nodes)), dtype=int)
        valid_mask[valid_samples_indices] = 1
        valid_mask = valid_mask.astype(bool)
        test_mask = np.zeros((len(nodes)), dtype=int)
        test_mask[test_samples_indices] = 1
        test_mask = test_mask.astype(bool)

        return train_mask, test_mask, valid_mask

    def prepare_training_data(self):
        for n, index in self.nodes.items():
            y_probabilty_vector = list(np.zeros((self.number_of_classes)))
            y_probabilty_vector[int(self.node_labels[index]) - 1] = 1.0
            self.y.append(y_probabilty_vector)


class UndirectedGraph(Graph):
    def __init__(self, edges, nodes=None):
        super().__init__(nodes, edges)
        self.adjacency = self.create_adjacency_matrix()
        self.degree_matrix = self.create_degree_matrix()
        self.degree_normalized_adjacency = self.degree_normalized_adjacency()
        self.out_adjacency_probability_test()

    def __str__(self):
        return "An undirected graph of " + str(self.N) + " nodes and " + str(self.number_of_edges) + " edges."

    def out_adjacency_probability_test(self):
        print(np.sum(self.adjacency, axis=1))

    def degree_normalized_adjacency(self):
        self_connections = self.adjacency + np.identity(self.N)
        inverse_degree = np.linalg.inv(self.degree_matrix)
        return np.array(np.multiply(self_connections, inverse_degree))

    def create_degree_matrix(self):
        return np.diag(np.count_nonzero(self.adjacency, axis=0))

    def create_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.N, self.N), dtype=float)
        for oes in self.edges:
            adjacency_matrix[oes[0], oes[1]] = oes[2]
            adjacency_matrix[oes[1], oes[0]] = oes[2]
        return adjacency_matrix


class DirectedGraph(Graph):
    def __init__(self, nodes, edges):
        super().__init__(nodes, edges)
        self.in_indexed_edges = self.flip_edges(self.edges)
        self.N = len(self.nodes.keys())
        self.out_adjacency_matrix, self.in_adjacency_matrix = self.create_adjacency_matrices()
        self.out_degree_matrix, self.in_degree_matrix = self.create_degree_matrices()
        self.normalized_out_adjacecny, self.normalized_in_adjacecny = self.normalize_directed_adjacencies()
        self.out_adjacency_probability_test()

    def __str__(self):
        return "An directed graph of " + str(self.N) + " nodes and " + str(self.number_of_edges) + " edges."

    def out_adjacency_probability_test(self):
        print(np.sum(self.out_adjacency_matrix, axis=1))

    def normalize_directed_adjacencies(self):
        return np.multiply(self.out_adjacency_matrix, np.linalg.inv(self.out_degree_matrix)), np.multiply(
            self.in_adjacency_matrix, np.linalg.inv(self.in_degree_matrix))

    def create_degree_matrices(self):
        return np.diag(np.count_nonzero(self.out_adjacency_matrix, axis=0)), np.diag(
            np.count_nonzero(self.in_adjacency_matrix, axis=0))

    def create_adjacency_matrices(self):
        in_adjacency_matrix = np.zeros((self.N, self.N), dtype=float)
        out_adjacency_matrix = np.zeros((self.N, self.N), dtype=float)
        for oes in self.edges:
            out_adjacency_matrix[oes[0], oes[1]] = oes[2]
        for ies in self.in_indexed_edges:
            in_adjacency_matrix[ies[0], ies[1]] = ies[2]
        return out_adjacency_matrix, in_adjacency_matrix

    @staticmethod
    def flip_edges(edge_list):
        flipped_edges = []
        for es in edge_list:
            flipped_edges.append((es[1], es[0], es[2]))
        return flipped_edges
