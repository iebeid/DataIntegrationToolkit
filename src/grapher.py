import numpy as np

from importer import *





class Graph:
    def __init__(self, edges=[], nodes={}):
        self.edges = edges
        self.nodes = nodes
        self.index_nodes()
        self.index_edges()
        # Check to see if all the nodes and all the edges match
        self.integrity_check()
        self.number_of_nodes = len(self.nodes.keys())
        self.number_of_edges = len(self.edges)

    def integrity_check(self):
        print("Graph integrity check")
        weightless_edges = []
        for e in self.edges:
            weightless_edges.append((e[0], e[1]))
        data = nx.Graph(weightless_edges)
        connected_components = nx.connected_components(data)
        if len(list(connected_components)) > 1:
            print("WARNING: Graph has more than one component")
        else:
            print("Graph has no components")
        all_nodes = []
        for e in self.edges:
            s_n = e[0]
            t_n = e[1]
            all_nodes.append(s_n)
            all_nodes.append(t_n)
        all_nodes_from_edges = sorted(list(set(all_nodes)))
        input_nodes = list(self.nodes.keys())
        input_nodes.sort()
        all_nodes_from_nodes = input_nodes
        if all_nodes_from_nodes != all_nodes_from_edges:
            print("WARNING: nodes and edges are not equal")
        else:
            print("Nodes matches edges")

    def index_nodes(self):
        self.node_index = {}
        self.node_inverted_index = {}
        if len(self.nodes) == 0:
            all_nodes = []
            for pair in self.edges:
                source_node = str(pair[0])
                target_node = str(pair[1])
                all_nodes.append(source_node)
                all_nodes.append(target_node)
            all_nodes = list(np.unique(np.array(all_nodes)))
            for i in range(len(all_nodes)):
                self.nodes[i] = "1"
                self.node_index[all_nodes[i]] = i
                self.node_inverted_index[i] = all_nodes[i]
        else:
            for i, n in enumerate(self.nodes.keys()):
                self.node_index[n] = i
                self.node_inverted_index[i] = n
            new_nodes = {}
            for n in self.nodes:
                new_nodes[self.node_index[n]] = self.nodes[n]
            self.nodes=new_nodes


    def index_edges(self):
        indexed_edge_list = []
        for es in self.edges:
            indexed_edge_list.append((self.node_index[es[0]], self.node_index[es[1]], es[2]))
        self.edges = indexed_edge_list

    def create_node_labels(self):
        self.node_labels = {}
        self.node_label_profile = {}
        self.number_of_classes = 0
        weightless_edges = []
        for e in self.edges:
            weightless_edges.append((e[0], e[1]))
        data = nx.Graph(weightless_edges)
        louvain_communities = nx.community.louvain_communities(data, seed=123)
        if len(list(louvain_communities)) < 2:
            print("WARNING: Graph has only one Louvain community")
        c = 0
        for community in louvain_communities:
            c = c + 1
            for n in community:
                self.node_labels[n] = str(c)
                self.nodes[n] = str(c)
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_labels.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)
        self.number_of_classes = int(len(self.node_label_profile.keys()))

    def balanced_node_label_sampler(self, split_percent):
        nodes = list(self.nodes.keys())
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
        self.y = []
        for n, label in self.nodes.items():
            y_probabilty_vector = list(np.zeros((self.number_of_classes)))
            y_probabilty_vector[int(int(label) - 1)] = 1.0
            self.y.append(y_probabilty_vector)

        self.y = tf.reshape(tf.convert_to_tensor(self.graph_object.y, dtype=tf.float32),
                            shape=(self.graph_object.n, self.graph_object.number_of_classes))
        self.X = tf.Variable(UtilityOperation.identity_initializer(self.graph_object.n, self.input_dim),
                             dtype=tf.float32)
        self.degree_normalized_adjacency = tf.constant(
            tf.convert_to_tensor(self.graph_object.degree_normalized_adjacency, dtype=tf.float32), dtype=tf.float32)


class BipartiteGraph(Graph):
    def __init__(self, edges=[], nodes={}):
        super().__init__(edges=edges, nodes=nodes)
        self.inverted_node_memberships = UtilityOperation.invert_dict(self.nodes)
        self.type1_nodes = list(list(self.inverted_node_memberships.values())[0])
        self.type2_nodes = list(list(self.inverted_node_memberships.values())[1])
        if len(list(self.inverted_node_memberships.values())) > 2:
            print("WARNING: Graph has more than 2 node types")
        self.type1_memberships = []
        self.type2_memberships = []
        for n in self.type1_nodes:
            self.type1_memberships.append(self.node_index[n])
        for n in self.type2_nodes:
            self.type2_memberships.append(self.node_index[n])
        # Dimensions of the matrix
        self.type2_count = len(self.type2_nodes)
        self.type1_count = len(self.type1_nodes)
        self.create_biadjacency()
        self.create_degree_matrix()
        self.create_degree_normalized_biadjacency()

    def create_biadjacency(self):
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        self.type1_index = {self.type1_nodes[i]: i for i in range(0, self.type1_count)}
        self.type2_index = {self.type2_nodes[i]: i for i in range(0, self.type2_count)}
        # Computes weighted biadjacency matrix.
        self.biadjacency = np.matrix(np.zeros((self.type2_count, self.type1_count)))
        for edge in self.edges:
            self.biadjacency[int(self.type2_index[edge[0]]), int(self.type1_index[edge[1]])] = float(edge[2])

    def create_degree_matrix(self):
        self.type2_degree = np.array(np.sum(self.biadjacency, axis=0))
        self.type1_degree = np.array(np.sum(self.biadjacency, axis=1))
        self.combined_degrees = np.matrix(self.type1_degree @ self.type2_degree)

    def create_degree_normalized_biadjacency(self):
        self.degree_normalized_biadjacency = np.divide(self.biadjacency, self.combined_degrees)

    def modularity_clustering(self):
        # Modified BRIMS algorithm
        # Computes sum of edges and bimodularity matrix.
        m = float(sum(self.type1_degree))
        B = self.biadjacency - (self.combined_degrees / m)
        # Computation of initial modularity matrix for tar and reg nodes from the membership dataframe.
        T_ed = list(zip([self.type2_index[j] for j in [i for i in self.type2_nodes]], self.type2_memberships))
        T0 = np.zeros((self.type2_count, self.n))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = list(zip([self.type1_index[j] for j in [i for i in self.type1_nodes]], self.type1_memberships))
        R0 = np.zeros((self.type1_count, self.n))
        for edge in R_ed:
            R0[edge] = 1
        deltaQmin = min(1 / m, 1e-5)
        Qnow = 0
        deltaQ = 1
        while (deltaQ > deltaQmin):
            # Right sweep
            Tp = T0.transpose().dot(B)
            R = np.zeros((self.type1_count, self.n))
            am = np.array(np.argmax(Tp.transpose(), axis=1))
            for i in range(0, len(am)):
                R[i, am[i][0]] = 1
            # Left sweep
            Rp = B.dot(R)
            T = np.zeros((self.type2_count, self.n))
            am = np.array(np.argmax(Rp, axis=1))
            for i in range(0, len(am)):
                T[i, am[i][0]] = 1
            T0 = T
            Qthen = Qnow
            RtBT = T.transpose().dot(B.dot(R))
            Qcoms = (1 / m) * (np.diagonal(RtBT))
            Qnow = sum(Qcoms)
            deltaQ = Qnow - Qthen

        # Character Communities
        tar_memberships = list(zip(list(self.type2_index), [T[i, :].argmax() for i in range(0, len(self.type2_index))]))
        type2_memberships = {}
        for tup in tar_memberships:
            type2_memberships[str(tup[0])] = int(tup[1])
        grouped_type2_memberships = defaultdict(list)
        for index, row in sorted(type2_memberships.items()):
            grouped_type2_memberships[row].append(index)
        type2_components = []
        for k, v in grouped_type2_memberships.items():
            type2_components.append(v)

        # Entity Communities
        entity_memberships = list(
            zip(list(self.type1_index), [R[i, :].argmax() for i in range(0, len(self.type1_index))]))
        type1_memberships = {}
        for tup in entity_memberships:
            type1_memberships[str(tup[0])] = int(tup[1])
        grouped_type1_memberships = defaultdict(list)
        for index, row in sorted(type1_memberships.items()):
            grouped_type1_memberships[row].append(index)
        type1_components = []
        for k, v in grouped_type1_memberships.items():
            type1_components.append(v)

        return type1_components, type2_components


class UndirectedGraph(Graph):
    def __init__(self, edges=[], nodes={}):
        super().__init__(edges=edges, nodes=nodes)
        self.adjacency = self.create_adjacency_matrix()
        self.weighted_adjacency = self.create_weighted_adjacency_matrix()
        self.degree_matrix = self.create_degree_matrix()
        self.weighted_degree_matrix = self.create_weighted_degree_matrix()
        self.degree_normalized_adjacency = self.degree_normalized_adjacency()
        self.degree_weighted_normalized_adjacency = self.degree_weighted_normalized_adjacency()
        # self.out_adjacency_probability_test()

    def __str__(self):
        return "An undirected graph of " + str(self.number_of_nodes) + " nodes and " + str(
            self.number_of_edges) + " edges."

    def out_adjacency_probability_test(self):
        print(np.sum(self.adjacency, axis=1))

    def create_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            s_n = int(oes[0])
            t_n = int(oes[1])
            adjacency_matrix[s_n, t_n] = 1
            adjacency_matrix[t_n, s_n] = 1
        return adjacency_matrix

    def create_weighted_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            s_n = int(oes[0])
            t_n = int(oes[1])
            w = float(oes[2])
            adjacency_matrix[s_n, t_n] = w
            adjacency_matrix[t_n, s_n] = w
        return adjacency_matrix

    def create_degree_matrix(self):
        return np.diag(np.count_nonzero(self.adjacency, axis=0))

    def create_weighted_degree_matrix(self):
        return np.diag(np.sum(self.weighted_adjacency, axis=0))

    def degree_normalized_adjacency(self):
        self_connections = self.adjacency + np.identity(self.number_of_nodes)
        inverse_degree = np.linalg.inv(self.degree_matrix)
        return np.array(np.multiply(self_connections, inverse_degree))

    def degree_weighted_normalized_adjacency(self):
        self_connections = self.weighted_adjacency + np.identity(self.number_of_nodes)
        inverse_degree = np.linalg.inv(self.weighted_degree_matrix)
        return np.array(np.multiply(self_connections, inverse_degree))


class DirectedGraph(Graph):
    def __init__(self, edges=[], nodes={}):
        super().__init__(edges=edges, nodes=nodes)
        self.in_indexed_edges = UtilityOperation.flip_list(self.edges)
        self.out_adjacency_matrix, self.in_adjacency_matrix = self.create_adjacency_matrices()
        self.out_weighted_adjacency_matrix, self.in_weighted_adjacency_matrix = self.create_weighted_adjacency_matrices()
        self.out_degree_matrix, self.in_degree_matrix = self.create_degree_matrices()
        self.out_weighted_degree_matrix, self.in_weighted_degree_matrix = self.create_weighted_degree_matrix()
        self.normalized_out_adjacecny, self.normalized_in_adjacecny = self.degree_normalized_directed_adjacencies()
        self.normalized_weighted_out_adjacecny, self.normalized_weighted_in_adjacecny = self.degree_normalized_weighted_directed_adjacencies()
        # self.out_adjacency_probability_test()

    def __str__(self):
        return "An directed graph of " + str(self.number_of_nodes) + " nodes and " + str(
            self.number_of_edges) + " edges."

    def create_adjacency_matrices(self):
        in_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        out_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            out_adjacency_matrix[oes[0], oes[1]] = 1
        for ies in self.in_indexed_edges:
            in_adjacency_matrix[ies[0], ies[1]] = 1
        return out_adjacency_matrix, in_adjacency_matrix

    def create_weighted_adjacency_matrices(self):
        in_weighted_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        out_weighted_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            out_weighted_adjacency_matrix[oes[0], oes[1]] = oes[2]
        for ies in self.in_indexed_edges:
            in_weighted_adjacency_matrix[ies[0], ies[1]] = ies[2]
        return out_weighted_adjacency_matrix, in_weighted_adjacency_matrix

    def create_degree_matrices(self):
        return np.diag(np.count_nonzero(self.out_adjacency_matrix, axis=0)), np.diag(
            np.count_nonzero(self.in_adjacency_matrix, axis=0))

    def create_weighted_degree_matrix(self):
        return np.diag(np.sum(self.out_weighted_adjacency_matrix, axis=0)), np.diag(
            np.sum(self.in_weighted_adjacency_matrix, axis=0))

    def out_adjacency_probability_test(self):
        print(np.sum(self.out_adjacency_matrix, axis=1))

    def degree_normalized_directed_adjacencies(self):
        return np.multiply(self.out_adjacency_matrix, np.linalg.inv(self.out_degree_matrix)), np.multiply(
            self.in_adjacency_matrix, np.linalg.inv(self.in_degree_matrix))

    def degree_normalized_weighted_directed_adjacencies(self):
        return np.multiply(self.out_weighted_adjacency_matrix,
                           np.linalg.inv(self.out_weighted_degree_matrix)), np.multiply(
            self.in_weighted_adjacency_matrix, np.linalg.inv(self.in_weighted_degree_matrix))

    def create_node_labels(self):
        self.node_labels = {}
        self.node_label_profile = {}
        self.number_of_classes = 0
        data = nx.DiGraph()
        data.add_weighted_edges_from(self.edges)
        louvain_communities = nx.community.louvain_communities(data, seed=123)
        if len(list(louvain_communities)) < 2:
            print("WARNING: Graph has only one Louvain community")
        c = 0
        for community in louvain_communities:
            c = c + 1
            for n in community:
                self.node_labels[n] = str(c)
                self.nodes[n] = str(c)
        self.node_label_profile = defaultdict(list)
        for key, val in sorted(self.node_labels.items()):
            self.node_label_profile[val].append(key)
        self.node_label_profile = dict(self.node_label_profile)
        self.number_of_classes = int(len(self.node_label_profile.keys()))
