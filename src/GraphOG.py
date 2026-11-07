from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse

from GraphOG_Algorithms import Algorithms
from GraphOG_Tasks import Tasks


class GraphOG:
    # Constructor
    def __init__(self, edges, nodes=None, weighted=True, undirected=True, link_single_nodes=False, simple_graph=False):
        if len(edges) == 0:
            print("Empty edge list")
        if weighted and undirected:
            # EDGES
            edges = list(set(edges))
            edges.sort(key=lambda x: x[2])
            self.edges = edges
            # NODES
            self.single_nodes = []
            unique_nodes = []
            for e in self.edges:
                unique_nodes.append(e[0])
                unique_nodes.append(e[1])
            self.unique_nodes = list(set(unique_nodes))
            if nodes is None:
                i = 0
                l = 0
                self.nodes = {}
                for n in self.unique_nodes:
                    self.nodes[str(n)] = {"index": int(i), "type": "type", "attribute": list([0]), "label": int(l)}
                    i = i + 1
                    l = l + 1
            else:
                self.nodes = dict(sorted(nodes.items()))
                passed_nodes = self.nodes.keys()
                self.single_nodes = list(set(passed_nodes) - set(self.unique_nodes))
                if (len(self.unique_nodes) < len(passed_nodes)) and link_single_nodes:
                    single_nodes = list(set.difference(set(passed_nodes), set(self.unique_nodes)))
                    for n in single_nodes:
                        self.edges.append((str(n), str(n), 1.0))
                if len(self.unique_nodes) > len(passed_nodes):
                    single_nodes = list(set.difference(set(passed_nodes), set(self.unique_nodes)))
                    for n in single_nodes:
                        for e in self.edges[:]:
                            s_n = e[0]
                            t_n = e[1]
                            w = e[2]
                            if n == s_n or n == t_n:
                                self.edges.remove((s_n, t_n, w))
            self.N = len(self.nodes.keys())
            self.node_attribute = {}
            self.node_label = {}
            self.node_index = {}
            self.node_type = {}
            self.node_inverted_index = {}
            self.node_label_profile = defaultdict(list)
            self.node_type_profile = defaultdict(list)
            self.node_label_count = {}
            self.node_type_count = {}
            self.node_real_id = {}
            node_index = 0
            for i, v in self.nodes.items():
                self.node_attribute[i] = v["attribute"]
                self.node_label[i] = v["label"]
                self.node_index[i] = node_index
                self.node_inverted_index[node_index] = i
                self.node_type[i] = v["type"]
                if nodes is not None:
                    self.node_real_id[i] = v["real_id"]
                node_index = node_index + 1
            for key, val in sorted(self.node_type.items()):
                self.node_type_profile[val].append(key)
            self.node_type_profile = dict(self.node_type_profile)
            for t, ns in self.node_type_profile.items():
                self.node_type_count[t] = len(ns)
            for key, val in sorted(self.node_label.items()):
                self.node_label_profile[val].append(key)
            self.node_label_profile = dict(self.node_label_profile)
            # ADJACENCY MATRIX
            row = []
            col = []
            data_ones = []
            data_weights = []
            for e in self.edges:
                row.append(self.node_index[e[0]])
                col.append(self.node_index[e[1]])

                data_weights.append(e[2])
                data_ones.append(1)
            adjacency_csr_1 = scipy.sparse.csc_matrix((np.array(data_ones), (np.array(row), np.array(col))),
                                                      shape=(self.N, self.N))
            adjacency_csr_weighted_1 = scipy.sparse.csc_matrix((np.array(data_weights), (np.array(row), np.array(col))),
                                                               shape=(self.N, self.N))
            adjacency_csr_2 = adjacency_csr_1.transpose()
            adjacency_csr_weighted_2 = adjacency_csr_weighted_1.transpose()
            self.adjacency_asymmetrical = adjacency_csr_1
            self.adjacency_weighted_asymmetrical = adjacency_csr_weighted_1
            self.adjacency = adjacency_csr_1 + adjacency_csr_2
            self.adjacency_enriched = self.adjacency + scipy.sparse.eye(self.N, self.N)
            self.adjacency_weighted = adjacency_csr_weighted_1 + adjacency_csr_weighted_2
            self.adjacency_enriched_weighted = self.adjacency_weighted + scipy.sparse.eye(self.N,
                                                                                          self.N)
            # OPERATIONS
            self.tasks = Tasks(self)
            self.algorithms = Algorithms(self)
            # NETWORKX
            g = nx.Graph()
            g.add_weighted_edges_from(self.edges)
            self.to_networkx = g
            if not simple_graph:
                # DEGREE MATRIX
                d_row = []
                d_col = []
                d_data = []
                d_data_weights = []
                for n in self.nodes.keys():
                    index = self.node_index[n]
                    d_row.append(index)
                    d_col.append(index)
                    d_data_weights.append(float(self.tasks.get_sum_of_neighborhood_weights(n)))
                    d_data.append(float(len(self.tasks.get_neighbors(n))))
                self.degree = scipy.sparse.csc_matrix((np.array(d_data), (np.array(d_row), np.array(d_col))),
                                                      shape=(self.N, self.N))
                self.degree_weighted = scipy.sparse.csc_matrix(
                    (np.array(d_data_weights), (np.array(d_row), np.array(d_col))),
                    shape=(self.N, self.N))
                # LAPLACIAN MATRIX
                self.laplacian = self.degree - self.adjacency
                self.laplacian_weighted = self.degree - self.adjacency_weighted
                self.laplacian_weighted_degree_weighted = self.degree_weighted - self.adjacency_weighted

    def info(self, log_file):
        print("GraphOG graph object")
        print("GraphOG graph object", file=log_file)
        print("Number of nodes: " + str(len(self.nodes.keys())))
        print("Number of nodes: " + str(len(self.nodes.keys())), file=log_file)
        print("Number of represented nodes: " + str(len(self.unique_nodes)))
        print("Number of represented nodes: " + str(len(self.unique_nodes)), file=log_file)
        print("Number of single nodes: " + str(len(self.single_nodes)))
        print("Number of single nodes: " + str(len(self.single_nodes)), file=log_file)
        print("Number of edges: " + str(len(self.edges)))
        print("Number of edges: " + str(len(self.edges)), file=log_file)
        final_modularity = self.tasks.compute_modularity()
        print("Modularity: " + str(final_modularity))
        print("Modularity: " + str(final_modularity), file=log_file)

    def update_node_labels_with_components(self, components):
        # convert pair list to node labels
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        clusters_dict = {}
        for c in cluster_list:
            clusters_dict[c[1]] = c[0]
        clusters_res = defaultdict(list)
        for key, val in sorted(clusters_dict.items()):
            clusters_res[val].append(key)
        for n in self.nodes.keys():
            self.node_label[n] = None
        i = 0
        for k, v in clusters_res.items():
            for n in v:
                self.node_label[n] = i
            i = i + 1
        k = i
        for n in self.nodes.keys():
            if self.node_label[n] is None:
                self.node_label[n] = k
                k = k + 1
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_label.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)

    def update_node_labels(self, new_node_labels):
        self.node_label = new_node_labels
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_label.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)

    def node_labels_to_components(self):
        components = []
        for cc in self.node_label_profile.values():
            components.append(cc)
        return components

    def filter_edges(self, threshold, link_single_nodes, simple_graph):
        filtered_edges = []
        for e in self.edges:
            w = e[2]
            if w > threshold:
                filtered_edges.append((e[0], e[1], w))
        return GraphOG(filtered_edges, nodes=self.nodes, undirected=True, weighted=True, link_single_nodes=link_single_nodes,
                       simple_graph=simple_graph)

    def subgraph(self, nodes, link_single_nodes, simple_graph):
        edges = []
        for n in nodes:
            neighbors = self.tasks.get_neighbors(n)
            for nn in neighbors:
                if nn not in nodes:
                    continue
                if n < nn:
                    edges.append((n, nn, self.tasks.get_edge_weight(n, nn)))
        vertices = {}
        l = 0
        for nn in nodes:
            vertices[str(nn)] = {"real_id": self.node_real_id[nn], "type": self.node_type[nn], "attribute": self.node_attribute[nn],
                                 "label": int(l)}
            l = l + 1
        return GraphOG(edges, nodes=vertices, undirected=True, weighted=True, link_single_nodes=link_single_nodes,
                       simple_graph=simple_graph)

    def recast_graph(self, new_edges, link_single_nodes, simple_graph):
        return GraphOG(new_edges, nodes=self.nodes, undirected=True, weighted=True, link_single_nodes=link_single_nodes,
                       simple_graph=simple_graph)

    def is_complete_and_full(self):
        number_of_edges = len(self.edges)
        complete_edges = (self.N * (self.N - 1)) / 2
        check = False
        if number_of_edges == complete_edges:
            s = 0
            for e in self.edges:
                s = s + e[2]
            if s == number_of_edges:
                check = True
        return check
