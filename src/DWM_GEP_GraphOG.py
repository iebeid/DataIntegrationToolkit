import sys
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse

from modules.GraphEntityProfiler.DWM_GEP_GraphOG_Algorithms import Algorithms
from modules.GraphEntityProfiler.DWM_GEP_GraphOG_Tasks import Tasks


class GraphOG:

    # Constructor
    def __init__(self, edges, nodes=None, weighted=True, undirected=True, link_single_nodes=False):
        if len(edges) == 0:
            print("Empty edge list")
            sys.exit()
        if weighted and undirected:
            # EDGES
            edges.sort()
            self.edges = edges
            # NODES
            nodes = dict(sorted(nodes.items()))
            self.nodes = nodes
            # process nodes and edges
            unique_nodes = []
            for e in self.edges:
                unique_nodes.append(e[0])
                unique_nodes.append(e[1])
            self.unique_nodes = list(set(unique_nodes))
            passed_nodes = self.nodes.keys()
            # if len(self.unique_nodes) < len(passed_nodes):
            self.single_nodes = list(set(passed_nodes) - set(self.unique_nodes))
            if self.nodes is None:
                i = 0
                l = 0
                self.nodes = {}
                for n in self.nodes:
                    self.nodes[str(n)] = {"index": int(i), "type": "type", "attribute": list([0]), "label": l}
                    i = i + 1
                    l = l + 1
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
            self.node_attribute = {}
            self.node_label = {}
            self.node_index = {}
            self.node_type = {}
            self.node_inverted_index = {}
            self.node_label_profile = defaultdict(list)
            self.node_type_profile = defaultdict(list)
            self.node_label_count = {}
            self.node_type_count = {}
            node_index = 0
            for i, v in self.nodes.items():
                self.node_attribute[i] = v["attribute"]
                self.node_label[i] = v["label"]
                self.node_index[i] = node_index
                self.node_inverted_index[node_index] = i
                self.node_type[i] = v["type"]
                node_index = node_index + 1
            for key, val in sorted(self.node_type.items()):
                self.node_type_profile[val].append(key)
            self.node_type_profile = dict(self.node_type_profile)
            for t, ns in self.node_type_profile.items():
                self.node_type_count[t] = len(ns)
            for key, val in sorted(self.node_label.items()):
                self.node_label_profile[val].append(key)
            self.node_label_profile = dict(self.node_label_profile)
            # EDGE LIST TO INDICES
            self.index_edges = []
            for e in self.edges:
                self.index_edges.append((self.node_index[e[0]],self.node_index[e[1]],e[2]))
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
            adjacency_csr_1 = scipy.sparse.csr_matrix((np.array(data_ones), (np.array(row), np.array(col))),
                                                      shape=(len(self.nodes.keys()), len(self.nodes.keys())))
            adjacency_csr_weighted_1 = scipy.sparse.csr_matrix((np.array(data_weights), (np.array(row), np.array(col))),
                                                               shape=(len(self.nodes.keys()), len(self.nodes.keys())))
            adjacency_csr_2 = adjacency_csr_1.transpose()
            adjacency_csr_weighted_2 = adjacency_csr_weighted_1.transpose()
            self.adjacency_asymmetrical = adjacency_csr_1
            self.adjacency_weighted_asymmetrical = adjacency_csr_weighted_1
            self.adjacency = adjacency_csr_1 + adjacency_csr_2
            self.adjacency_enriched = self.adjacency + scipy.sparse.eye(len(self.nodes.keys()), len(self.nodes.keys()))
            self.adjacency_weighted = adjacency_csr_weighted_1 + adjacency_csr_weighted_2
            self.adjacency_enriched_weighted = self.adjacency_weighted + scipy.sparse.eye(len(self.nodes.keys()),
                                                                                          len(self.nodes.keys()))
            # OPERATIONS
            self.tasks = Tasks(self)
            self.algorithms = Algorithms(self)
            # EVALUATION
            self.link_index_pairs = []
            self.link_index_dict = {}
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
            self.degree = scipy.sparse.csr_matrix((np.array(d_data), (np.array(d_row), np.array(d_col))),
                                                  shape=(len(self.nodes.keys()), len(self.nodes.keys())))
            self.degree_weighted = scipy.sparse.csr_matrix(
                (np.array(d_data_weights), (np.array(d_row), np.array(d_col))),
                shape=(len(self.nodes.keys()), len(self.nodes.keys())))
            self.degree_weighted_normalized = scipy.sparse.csr_matrix(
                np.true_divide(self.degree_weighted.todense(), self.degree.todense(),
                               where=(self.degree_weighted.todense() != 0) | (
                                       self.degree.todense() != 0)))
            # LAPLACIAN MATRIX
            self.laplacian = self.degree - self.adjacency
            self.laplacian_weighted = self.degree - self.adjacency_weighted
            self.laplacian_weighted_degree_weighted = self.degree_weighted - self.adjacency_weighted
            # NETWORKX
            g = nx.Graph()
            g.add_weighted_edges_from(self.edges)
            self.to_networkx = g

    def info(self):
        print("GraphOG graph object")
        print("Number of nodes: " + str(len(self.nodes.keys())))
        print("Number of represented nodes: " + str(len(self.unique_nodes)))
        print("Number of single nodes: " + str(len(self.single_nodes)))
        print("Number of edges: " + str(len(self.edges)))

    def update_node_labels(self, clusters_list):
        # convert pair list to node labels
        clusters_dict = {}
        for c in clusters_list:
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

    def convert_node_labels_to_link_index(self):
        for c in self.node_label_profile.values():
            minimum_node = min(c)
            for n in c:
                self.link_index_pairs.append((minimum_node, n))
                self.link_index_dict[n] = minimum_node


    def subgraph(self, nodes, link_single_nodes):
        edges = []
        if len(nodes) > 1:
            if len(self.edges) > 0:
                for n in nodes:
                    for e in self.edges:
                        n1 = e[0]
                        n2 = e[1]
                        w = e[2]
                        if (n == n1 and n1 in nodes) or (n == n2 and n2 in nodes):
                            edges.append((n1, n2, w))
            else:
                for n in nodes:
                    edges.append((n, n, 1.0))
        elif len(nodes) == 1:
            for n in nodes:
                edges.append((n, n, 1.0))
        edges = list(set(edges))

        new_nodes = []
        for e in edges:
            new_nodes.append(e[0])
            new_nodes.append(e[1])
        new_nodes = list(set(new_nodes))

        vertices = {}
        l = 0
        for nn in new_nodes:
            vertices[str(nn)] = {"type": self.node_type[nn], "attribute": self.node_attribute[nn],
                                 "label": int(l)}
            l = l + 1
        return GraphOG(edges, nodes=vertices, undirected=True, weighted=True, link_single_nodes=link_single_nodes)

    def recast_graph(self, new_edges):
        return GraphOG(new_edges, nodes=self.nodes, undirected=True, weighted=True, link_single_nodes=False)
