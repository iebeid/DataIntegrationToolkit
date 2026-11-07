import itertools
import random
from collections import defaultdict
from collections import deque

import community as louvain
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.stats
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


class Algorithms():

    # Constructor
    def __init__(self, graph):
        self.graph = graph

    # Kolb, L., Sehili, Z., & Rahm, E. (2014). Iterative computation of connected graph
    # components with MapReduce. Datenbank-Spektrum, 14(2), 107-117.
    def talburt_transitive_closure_connected_component_detection(self):
        pairList = self.graph.edges
        # Bootstap process by add reverse of all pairs to the pairList
        iterationCnt = 0
        clusterList = []
        for pair in pairList:
            clusterList.append((pair[0], pair[1]))
            pairRev = (pair[1], pair[0])
            clusterList.append(pairRev)
            pairSelf1 = (pair[0], pair[0])
            clusterList.append(pairSelf1)
            pairSelf2 = (pair[1], pair[1])
            clusterList.append(pairSelf2)
        # Change 2.10
        pairList = list(set(clusterList))
        # Sort pairs in order by the first position (the key)
        pairList.sort()
        # All of the pairs with same key are a Key Group
        clusterList = []
        moreWorkToDo = True
        iteration = 0
        while moreWorkToDo:
            moreWorkToDo = False
            iteration += 1
            # Add a caboose record to the end of the pairList
            caboose = ('---', '---')
            pairList.append(caboose)
            keyGroup = []
            for j in range(0, len(pairList) - 1):
                currentPair = pairList[j]
                keyGroup.append(currentPair)
                # Look ahead to the next key
                nextPair = pairList[j + 1]
                currentKey = currentPair[0]
                nextKey = nextPair[0]
                # When next key is different, at end of Key Group and ready to process keyGroup
                if currentKey != nextKey:
                    firstGroupPair = keyGroup[0]
                    firstGroupPairKey = firstGroupPair[0]
                    firstGroupPairValue = firstGroupPair[1]
                    # Add new pairs to clusterList from key groups starting with reversed pair and larger than 1 pair
                    keyGroupSize = len(keyGroup)
                    if firstGroupPairKey > firstGroupPairValue:
                        if keyGroupSize > 1:
                            moreWorkToDo = True
                            for k in range(keyGroupSize):
                                groupPair = keyGroup[k]
                                groupPairValue = groupPair[1]
                                newPair = (firstGroupPairValue, groupPairValue)
                                clusterList.append(newPair)
                                newReversePair = (groupPairValue, firstGroupPairValue)
                                clusterList.append(newReversePair)
                            # Decide if first pair of keyGroup should move over to clusterList
                            lastGroupPair = keyGroup[keyGroupSize - 1]
                            lastGroupPairValue = lastGroupPair[1]
                            if firstGroupPairKey < lastGroupPairValue:
                                clusterList.append(firstGroupPair)
                    else:
                        # pass other key groups forward to cluster list
                        clusterList.extend(keyGroup)
                    keyGroup = []
            pairList = []
            # Change 2.10
            pairList = list(set(clusterList))
            pairList.sort()
            iterationCnt += 1
            clusterList = []
        cluster_ids = {}
        for p in pairList:
            cluster_ids[p[1]] = p[0]
        node_label_profile = defaultdict(list)
        for key, val in sorted(cluster_ids.items()):
            node_label_profile[val].append(key)
        node_label_profile = dict(node_label_profile)
        components = []
        for v in node_label_profile.values():
            components.append(v)
        return components

    # D. J. Pearce, “An Improved Algorithm for Finding the Strongly Connected
    # Components of a Directed Graph”, Technical Report, 2005
    def scipy_depth_first_connected_component_detection(self):
        n_components, labels = connected_components(csgraph=self.graph.adjacency, directed=False,
                                                    return_labels=True)
        components = []
        membership = {}
        for i, l in enumerate(labels):
            membership[self.graph.node_inverted_index[i]] = l
        res = defaultdict(list)
        for key, val in sorted(membership.items()):
            res[val].append(key)
        for k, v in res.items():
            components.append(v)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Nuutila, E., & Soisalon-Soininen, E. (1994). On finding the strongly connected components
    # in a directed graph. Information processing letters, 49(1), 9-14.
    def networkx_breadth_first_connected_component_detection(self):
        g = self.graph.to_networkx
        components = nx.connected_components(g)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Simple naive breadth first traversal of a graph
    def __breadth_first_search(self, root):
        visited = set()
        visited.add(root)
        queue = deque([root])
        while queue:
            dequeued_node = queue.popleft()
            neighbors = self.graph.tasks.get_neighbors(dequeued_node)
            for neighbour in neighbors:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.extend(neighbors)
        return list(visited)

    def akef_breadth_first_connected_component_detection(self):
        seen = set()
        components = []
        for n in self.graph.nodes.keys():
            if n not in seen:
                c = self.__breadth_first_search(n)
                seen.update(c)
                components.append(c)
        return components

    # QUICK CONNECTED COMPONENT DETECTION
    def akef_quick_connected_components_detection(self):
        components = []
        for ni_name, ni in self.graph.node_index.items():
            neighbor_indices_i = set(self.graph.adjacency[:, ni].indices)
            neighbor_indices_i.add(ni)
            intersections = []
            if len(neighbor_indices_i) > 1:
                for nj_name, nj in self.graph.node_index.items():
                    if ni != nj:
                        neighbor_indices_j = set(self.graph.adjacency[:, nj].indices)
                        neighbor_indices_j.add(nj)
                        if len(neighbor_indices_j) > 1:
                            if len(set.intersection(neighbor_indices_i, neighbor_indices_j)) != 0:
                                new_component = list(set.union(neighbor_indices_i, neighbor_indices_j))
                                intersections.append(new_component)
            component = list(set(list(itertools.chain.from_iterable(intersections))))
            component.sort()
            if len(component) > 0:
                components.append(component)
        res = []
        [res.append(x) for x in components if x not in res]
        cluster_list = []
        for cc in res:
            cc_resolved = []
            for n in cc:
                cc_resolved.append(self.graph.node_inverted_index[n])
            least_node = min(cc_resolved)
            for n in cc_resolved:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    def akef_unbiased_random_walker(self, walk_length):
        walks = []
        for node in tqdm(self.graph.nodes.keys()):
            neighbours = self.graph.tasks.get_neighbors(node)
            if neighbours:
                number_of_walks_per_node = len(neighbours)
                for _ in range(number_of_walks_per_node):
                    walk = [node]
                    current_node = node
                    for _ in range(walk_length - 1):
                        neighbours = self.graph.tasks.get_neighbors(current_node)
                        if len(neighbours) == 0:
                            # dead end, so stop
                            break
                        else:
                            # has neighbours, so pick one to walk to
                            current_node = random.choice(neighbours)
                        walk.append(current_node)
                    walks.append(walk)
                    if walk:
                        walks.append(walk)
            else:
                walks.append([node])
        return walks

    def akef_weighted_biased_random_walker(self, walk_length):
        walks = []
        for node in self.graph.nodes.keys():
            neighbours = self.graph.tasks.get_neighbors(node)
            if len(neighbours) > 0:
                number_of_walks_per_node = len(neighbours)
                for _ in range(number_of_walks_per_node):
                    walk = [node]
                    current_node = node
                    for _ in range(walk_length - 1):
                        neighbours = self.graph.tasks.get_neighbors(current_node)
                        neighbours_weights = self.graph.tasks.get_neighborhood_weights(current_node)
                        s = self.graph.tasks.get_sum_of_neighborhood_weights(current_node)
                        p_v = []
                        for n_w in neighbours_weights:
                            p_v.append(n_w / s)
                        current_node = np.random.choice(neighbours, p=p_v)
                        walk.append(current_node)
                    walks.append(walk)
                    if walk:
                        walks.append(walk)
            else:
                walks.append([node])
        return walks

    # Louvain method of commnuity detection using modularity optimization based on:
    # Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large
    # networks. Journal of statistical mechanics: theory and experiment, 2008(10), P10008.
    def networkx_louvain_community_detection(self):
        partitions = louvain.best_partition(self.graph.to_networkx, partition=self.graph.node_label)
        partitions_profiles = defaultdict(list)
        for n, com in sorted(partitions.items()):
            partitions_profiles[com].append(n)
        partitions_profiles = dict(partitions_profiles)
        components = []
        for k, v in partitions_profiles.items():
            components.append(v)
        self.graph.update_node_labels_with_components(components)
        return components

    def akef_modularity_maximizer(self):
        for i in self.graph.nodes.keys():
            # what if i removed i from its community
            i_com = self.graph.tasks.get_node_label(i)
            i_modularity = self.graph.tasks.compute_delta_modularity_node_remove(i_com, i)
            all_possible_moves = []
            all_possible_coms = []
            all_weights = []
            neighbors = self.graph.tasks.get_neighbors(i)
            if len(neighbors) > 0:
                for j in neighbors:
                    if i != j:
                        w = self.graph.tasks.get_edge_weight(i, j)
                        j_com = self.graph.tasks.get_node_label(j)
                        j_modularity = self.graph.tasks.compute_delta_modularity_node_add(j_com, i)
                        delta_modularity = float(i_modularity + j_modularity)
                        all_possible_moves.append(delta_modularity)
                        all_possible_coms.append(j_com)
                        all_weights.append(w)
                max_node_index = np.argmax(np.array(all_possible_moves))
                max_delta_mod_value = all_possible_moves[max_node_index]
                max_com = all_possible_coms[max_node_index]
                max_weight = all_weights[max_node_index]
                if (max_delta_mod_value > 0.0) and max_weight == 1:
                # if (max_delta_mod_value > 0.0):
                    self.graph.tasks.set_node_label(max_com, i)
        self.graph.update_node_labels(self.graph.node_label)
        components = self.graph.node_labels_to_components()
        return components

    def akef_greedy_modularity_optimizer(self):
        node_label_changes = self.graph.node_label
        final_label_changes = self.graph.node_label
        final_label_profile = self.graph.node_label_profile
        initial_modularity = self.graph.tasks.compute_modularity()
        previous_modularity = initial_modularity
        for i in self.graph.nodes.keys():
            i_com = self.graph.tasks.get_node_label(i)
            # remove i from its community and assign it to a single community
            node_label_changes[i] = int(max(list(final_label_profile.keys())) + 1)
            self.graph.update_node_labels(node_label_changes)
            current_modularity = self.graph.tasks.compute_modularity()
            delta_modularity_1 = current_modularity - previous_modularity
            previous_modularity = current_modularity
            # add i to community of j
            neighborhood = self.graph.tasks.get_neighbors(i)
            candidate_memberships = {}
            for j in neighborhood:
                j_com = self.graph.tasks.get_node_label(j)
                node_label_changes[i] = j_com
                self.graph.update_node_labels(node_label_changes)
                current_modularity = self.graph.tasks.compute_modularity()
                delta_modularity = delta_modularity_1 + (current_modularity - previous_modularity)
                candidate_memberships[j_com] = round(delta_modularity, 3)
            max_com = max(candidate_memberships, key=candidate_memberships.get)
            largest_delta_modularity_value = candidate_memberships[max_com]
            if largest_delta_modularity_value > 0:
                final_label_changes[i] = max_com
            if largest_delta_modularity_value < 0:
                final_label_changes[i] = int(max(list(final_label_profile.keys())) + 1)
            if largest_delta_modularity_value == 0:
                final_label_changes[i] = i_com
            self.graph.update_node_labels(final_label_changes)
            previous_modularity = self.graph.tasks.compute_modularity()
        final_modularity = self.graph.tasks.compute_modularity()
        print(self.graph.node_label_profile)
        print("initial_modularity:" + str(initial_modularity))
        print("final_modularity:" + str(final_modularity))

    # Barber, M. J. (2007). Modularity and community detection in bipartite networks. Physical Review E, 76(6), 066102.
    def akef_modified_brim_bipartite_cluster_detection(self):
        edges = []
        words = []
        documents = []
        for n in self.graph.nodes.keys():
            extracted_words = self.graph.node_attribute[n][0]
            documents.append(n)
            for t in extracted_words:
                edges.append((t, n))
                words.append(t)
        words = list(np.unique(np.array(words)))
        # ---------
        word_membership = []
        document_membership = []
        all_nodes = documents + words
        for mem, n in enumerate(all_nodes):
            if n in words:
                word_membership.append(mem)
            if n in documents:
                document_membership.append(mem)
        # Dimensions of the matrix
        p = len(documents)
        q = len(words)
        # c = doc_com
        c = len(all_nodes)
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        rg = {words[i]: i for i in range(0, q)}
        gn = {documents[i]: i for i in range(0, p)}
        # Computes weighted biadjacency matrix.
        A = np.matrix(np.zeros((p, q)))
        for edge in edges:
            A[gn[edge[1]], rg[edge[0]]] = 1.0
        # Computes node degrees for the nodesets.
        ki = A.sum(1)
        dj = A.sum(0)
        combined_degrees = ki @ dj
        # Computes sum of edges and bimodularity matrix.
        m = float(sum(ki))
        B = A - (combined_degrees / m)
        # Computation of initial modularity matrix for tar and reg nodes from the membership dataframe.
        T_ed = zip([gn[j] for j in [i for i in documents]], document_membership)
        T0 = np.zeros((p, c))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = zip([rg[j] for j in [i for i in words]], word_membership)
        R0 = np.zeros((q, c))
        for edge in R_ed:
            R0[edge] = 1
        deltaQmin = min(1 / m, 1e-5)
        Qnow = 0
        deltaQ = 1
        p, q = B.shape
        while (deltaQ > deltaQmin):
            # Right sweep
            Tp = T0.transpose().dot(B)
            R = np.zeros((q, c))
            am = np.array(np.argmax(Tp.transpose(), axis=1))
            for i in range(0, len(am)):
                R[i, am[i][0]] = 1
            # Left sweep
            Rp = B.dot(R)
            T = np.zeros((p, c))
            am = np.array(np.argmax(Rp, axis=1))
            for i in range(0, len(am)):
                T[i, am[i][0]] = 1
            T0 = T
            Qthen = Qnow
            RtBT = T.transpose().dot(B.dot(R))
            Qcoms = (1 / m) * (np.diagonal(RtBT))
            Qnow = sum(Qcoms)
            deltaQ = Qnow - Qthen
        tar_membership = list(zip(list(gn), [T[i, :].argmax() for i in range(0, len(gn))]))
        document_memberships = {}
        for tup in tar_membership:
            document_memberships[str(tup[0])] = int(tup[1])
        grouped_document_memberships = defaultdict(list)
        for index, row in sorted(document_memberships.items()):
            grouped_document_memberships[row].append(index)
        components = []
        for k, v in grouped_document_memberships.items():
            components.append(v)
        self.graph.update_node_labels_with_components(components)
        return components

    def akef_modified_brim_bipartite_cluster_detection_2(self):
        edges = []
        words = []
        documents = []
        document_membership = []
        for n in self.graph.nodes.keys():
            extracted_words = self.graph.node_attribute[n][0]
            document_membership.append(self.graph.node_label[n])
            documents.append(n)
            for t in extracted_words:
                edges.append((t, n))
                words.append(t)
        words = list(np.unique(np.array(words)))
        all_nodes = documents + words

        word_membership = []
        max_doc_com = np.max(np.array(list(self.graph.node_label.values())))
        for mem, n in enumerate(words):
            word_membership.append(mem+max_doc_com+1)

        # Dimensions of the matrix
        p = len(documents)
        q = len(words)
        # c = doc_com
        c = len(all_nodes)
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        rg = {words[i]: i for i in range(0, q)}
        gn = {documents[i]: i for i in range(0, p)}
        # Computes weighted biadjacency matrix.
        A = np.matrix(np.zeros((p, q)))
        for edge in edges:
            A[gn[edge[1]], rg[edge[0]]] = 1.0
        # Computes node degrees for the nodesets.
        ki = A.sum(1)
        dj = A.sum(0)
        combined_degrees = ki @ dj
        # Computes sum of edges and bimodularity matrix.
        m = float(sum(ki))
        B = A - (combined_degrees / m)
        # Computation of initial modularity matrix for tar and reg nodes from the membership dataframe.
        T_ed = zip([gn[j] for j in [i for i in documents]], document_membership)
        T0 = np.zeros((p, c))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = zip([rg[j] for j in [i for i in words]], word_membership)
        R0 = np.zeros((q, c))
        for edge in R_ed:
            R0[edge] = 1
        deltaQmin = min(1 / m, 1e-5)
        Qnow = 0
        deltaQ = 1
        p, q = B.shape
        while (deltaQ > deltaQmin):
            # Right sweep
            Tp = T0.transpose().dot(B)
            R = np.zeros((q, c))
            am = np.array(np.argmax(Tp.transpose(), axis=1))
            for i in range(0, len(am)):
                R[i, am[i][0]] = 1
            # Left sweep
            Rp = B.dot(R)
            T = np.zeros((p, c))
            am = np.array(np.argmax(Rp, axis=1))
            for i in range(0, len(am)):
                T[i, am[i][0]] = 1
            T0 = T
            Qthen = Qnow
            RtBT = T.transpose().dot(B.dot(R))
            Qcoms = (1 / m) * (np.diagonal(RtBT))
            Qnow = sum(Qcoms)
            deltaQ = Qnow - Qthen
        tar_membership = list(zip(list(gn), [T[i, :].argmax() for i in range(0, len(gn))]))
        document_memberships = {}
        for tup in tar_membership:
            document_memberships[str(tup[0])] = int(tup[1])
        grouped_document_memberships = defaultdict(list)
        for index, row in sorted(document_memberships.items()):
            grouped_document_memberships[row].append(index)
        components = []
        for k, v in grouped_document_memberships.items():
            components.append(v)
        self.graph.update_node_labels_with_components(components)
        return components


    # SPEK
    def spectral_embedding_method(self, id2refid):
        eigenvals, eigenvecs = scipy.linalg.eig(self.graph.laplacian_weighted.todense())
        print(eigenvals.shape)
        print(eigenvecs.shape)
        eigenvecs = np.array(eigenvecs.real)
        eigenvals = np.array(eigenvals.real)
        eigenvals_sorted = np.sort(eigenvals)
        eigenvecs_transposed = np.transpose(eigenvecs)
        fiedler_vectors = []
        for ev in range(0, len(eigenvals_sorted)):
            corresponding_eigenval_location = eigenvals.tolist().index(eigenvals_sorted[ev])
            corresponding_eigenval = eigenvals[corresponding_eigenval_location]
            scaled_eigenvecs = eigenvecs_transposed[corresponding_eigenval_location] * corresponding_eigenval
            fiedler_vectors.append(scaled_eigenvecs)
        X = np.array(fiedler_vectors).T
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, affinity="euclidean",
                                             compute_full_tree=True, compute_distances=True,
                                             connectivity=self.graph.adjacency.toarray(), linkage="single").fit(X)
        print(clustering.labels_)
        record_labels = {}
        for (row, label) in enumerate(clustering.labels_):
            print("row %s has label %d" % (id2refid[row], label))
            record_labels[id2refid[row]] = label
        node_label_profile = defaultdict(list)
        for key, val in sorted(record_labels.items()):
            node_label_profile[val].append(key)
        node_label_profile = dict(node_label_profile)
        return node_label_profile

    # Fiedler, M. (1975). A property of eigenvectors of nonnegative symmetric matrices and
    # its application to graph theory. Czechoslovak mathematical journal, 25(4), 619-633.
    def akef_spectral_unsigned_fiedler_connected_component_detection(self):
        components = []
        eigenvals, eigenvecs = scipy.linalg.eigh(self.graph.laplacian_weighted_enriched.todense())
        eigenvecs = eigenvecs.real
        eigenvals = eigenvals.real
        fiedler_pos = np.where(eigenvals == np.sort(eigenvals)[1])[0][0]
        fiedler_vector = np.real(np.transpose(eigenvecs)[fiedler_pos])
        positive_values = []
        negative_values = []
        neutral_values = []
        for j, v in enumerate(fiedler_vector):
            if float(v) > float(0):
                positive_values.append(self.graph.node_inverted_index[j])
            if float(v) < float(0):
                negative_values.append(self.graph.node_inverted_index[j])
            if float(v) == float(0):
                neutral_values.append(self.graph.node_inverted_index[j])
        components.append(positive_values)
        components.append(negative_values)
        components.append(neutral_values)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Fiedler, M. (1975). A property of eigenvectors of nonnegative symmetric matrices and
    # its application to graph theory. Czechoslovak mathematical journal, 25(4), 619-633.
    def akef_spectral_fiedler_connected_component_detection(self):
        components = []
        eigenvals, eigenvecs = scipy.linalg.eigh(self.graph.laplacian_weighted_enriched.todense())
        eigenvecs = eigenvecs.real
        eigenvals = eigenvals.real
        fiedler_pos = np.where(eigenvals == np.sort(eigenvals)[1])[0][0]
        fiedler_vector = np.real(np.transpose(eigenvecs)[fiedler_pos])
        fiedler_dict = {}
        for j, v in enumerate(fiedler_vector):
            fiedler_dict[self.graph.node_inverted_index[j]] = round(v, 10)
        fielder_grouped_sorted = defaultdict(list)
        for key, val in sorted(fiedler_dict.items()):
            fielder_grouped_sorted[val].append(key)
        for key, val in fielder_grouped_sorted.items():
            components.append(val)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Fiedler, M. (1975). A property of eigenvectors of nonnegative symmetric matrices and
    # its application to graph theory. Czechoslovak mathematical journal, 25(4), 619-633.
    def akef_spectral_signed_fiedler_connected_component_detection(self):
        components = []
        eigenvals, eigenvecs = scipy.linalg.eigh(self.graph.laplacian_weighted_enriched.todense())
        eigenvecs = eigenvecs.real
        eigenvals = eigenvals.real
        fiedler_pos = np.where(eigenvals == np.sort(eigenvals)[1])[0][0]
        fiedler_vector = np.real(np.transpose(eigenvecs)[fiedler_pos])
        positive_values = {}
        negative_values = {}
        neutral_values = {}
        for j, v in enumerate(fiedler_vector):
            if float(v) > float(0):
                positive_values[self.graph.node_inverted_index[j]] = round(v, 5)
            if float(v) < float(0):
                negative_values[self.graph.node_inverted_index[j]] = round(v, 5)
            if float(v) == float(0):
                neutral_values[self.graph.node_inverted_index[j]] = round(v, 5)
        positive_reorganized = defaultdict(list)
        for key, val in sorted(positive_values.items()):
            positive_reorganized[val].append(key)
        negative_reorganized = defaultdict(list)
        for key, val in sorted(negative_values.items()):
            negative_reorganized[val].append(key)
        neutral_reorganized = defaultdict(list)
        for key, val in sorted(neutral_values.items()):
            neutral_reorganized[val].append(key)
        for k, v in positive_reorganized.items():
            components.append(v)
        for k, v in negative_reorganized.items():
            components.append(v)
        for k, v in neutral_reorganized.items():
            components.append(v)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list
