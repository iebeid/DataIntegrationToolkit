from collections import defaultdict
from collections import deque
import textdistance
import community as louvain
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.stats
from scipy.sparse.csgraph import connected_components
import modules.DWM10_Parms as DWM10_Parms
import modules.DWM65_ScoringMatrixStd as DWM65_ScoringMatrixStd
import modules.DWM69_HierarchicalDocumentComparatorWithDigitDiscriminator as DWM69_HierarchicalDocumentComparatorWithDigitDiscriminator
import modules.GraphEntityProfiler.DWM_GEP_GraphOG_Math as DWM_GEP_GraphOG_Math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import random
from operator import itemgetter
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
            clusterList.append(pair)
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
        return pairList

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

    # Louvain method of commnuity detection using modularity optimization based on:
    # Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large
    # networks. Journal of statistical mechanics: theory and experiment, 2008(10), P10008.
    def networkx_louvain_community_detection(self):
        partitions = louvain.best_partition(self.graph.to_networkx)
        partitions_profiles = defaultdict(list)
        for n, com in sorted(partitions.items()):
            partitions_profiles[com].append(n)
        partitions_profiles = dict(partitions_profiles)
        components = []
        for k, v in partitions_profiles.items():
            components.append(v)
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
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # QUICK CONNECTED COMPONENT DETECTION


    # sliding window
    def __window(self, fseq, window):
        for i in range(len(fseq)-window+1):
            yield fseq[i:i+window]


    def akef_quick_connected_components_detection(self):
        current_indices = np.array([])
        previous_indices = np.array([])
        for ni in self.graph.node_index.values():
            node_column = self.graph.adjacency_enriched.toarray()[:][ni]
            neighbor_indices = np.array(np.where(node_column != 0)[0])
            current_indices = neighbor_indices

            previous_indices = current_indices


    def akef_quick_connected_components_detection(self):

        potential_components = []
        for ni in self.graph.node_index.values():
            node_column = self.graph.adjacency_enriched.toarray()[:][ni]
            neighbor_indices = np.where(node_column != 0)[0]
            potential_components.append(list(neighbor_indices))
        # potential_components = list(np.unique(np.array(potential_components,dtype=object)))
        # components = []

        for pc1 in potential_components[:]:
            possible_component = set(pc1)
            for pc2 in potential_components[:]:
                target_component = set(pc2)
                if (possible_component == target_component) or (not set.isdisjoint(possible_component,target_component)) or (not set.isdisjoint(target_component,possible_component)):
                    potential_components.remove(list(target_component))
                    potential_components.remove(list(possible_component))
                    possible_component.update(target_component)
                    potential_components.append(possible_component)
            # possible_component = list(np.unique(np.array(possible_component)))
            # components.append(possible_component)
        components = []
        return components





    # # sliding window
    # def __window(self, fseq, window):
    #     for i in range(len(fseq)-window+1):
    #         yield fseq[i:i+window]
    # def akef_quick_connected_components_detection(self, input_components):
    #     input_filtered_components = []
    #     for c in input_components:
    #         input_filtered_components.append((c[0],c[1]))
    #     overlapping_window_edges = self.__window(input_filtered_components,2)
    #     # cut = 0
    #     components = []
    #     # while cut != 0:
    #     component = set()
    #     for w in overlapping_window_edges:
    #         s1 = set(w[0])
    #         s2 = set(w[1])
    #         union = set(s1 | s2)
    #         intersection = set(s1 & s2)
    #         if len(intersection) !=0:
    #             component.update(union)
    #         else:
    #             if len(component)!=0:
    #                 components.append(component)
    #             else:
    #                 for n in union:
    #                     in_component = True
    #                     for cc in components:
    #                         if n not in cc:
    #                             in_component = False
    #                     if in_component is False:
    #                         components.append(set([n]))
    #             component = set()
    #     #merge components
    #     for cc1 in components.copy():
    #         for cc2 in components.copy():
    #             if cc1 != cc2:
    #                 if len(set(cc1 & cc2))!=0:
    #                     if cc1 in components:
    #                         components.remove(cc1)
    #                     if cc2 in components:
    #                         components.remove(cc2)
    #                     merged_component = set.union(cc1,cc2)
    #                     components.append(merged_component)
    #     return components







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


    def __get_next_node_with_max_modularity(self,i, number_of_walks_per_node):
        i_com = self.graph.node_label[i]
        delta_modularity_1 = self.graph.tasks.compute_delta_modularity(i_com, i)
        neighborhood = self.graph.tasks.get_neighbors(i)
        candidates = {}
        for j in neighborhood:
            edge_weight = self.graph.tasks.get_edge_weight(i,j)
            j_com = self.graph.node_label[j]
            delta_modularity = delta_modularity_1 + self.graph.tasks.compute_delta_modularity(j_com, i)
            weighted_delta_modularity = delta_modularity * edge_weight
            if weighted_delta_modularity > 0.0001:
                candidates[j] = weighted_delta_modularity
        candidate_nodes = []
        if candidates:
            sorted_candidates = {k: v for k,v in sorted(candidates.items(), key=itemgetter(1), reverse=True)}
            candidate_nodes = list(sorted_candidates.keys())[0:number_of_walks_per_node]
        return candidate_nodes


    def akef_hierarchical_modularity_graph_embedding_for_record_clustering(self):
        walk_length = 5
        number_of_walks_per_node = 1
        for i in self.graph.nodes.keys():
            walk = [i]
            previous_node = [i]
            for _ in range(walk_length):
                current_node = self.__get_next_node_with_max_modularity(previous_node[0],number_of_walks_per_node)
                if current_node:
                    for n in current_node:
                        walk.append(n)
                else:
                    break
                previous_node = current_node
            print(walk)



    # Greedy modularity optimization method
    def akef_greedy_modularity_optimization_community_detection(self):
        node_label_changes = self.graph.node_label
        for i, i_com in self.graph.node_label.items():
            neighborhood = self.graph.tasks.get_neighbors(i)
            candidate_memberships = {}
            delta_modularity_1 = self.graph.tasks.compute_delta_modularity(i_com, i)
            if delta_modularity_1 < 0:
                for j in neighborhood:
                # for j, j_com in self.graph.node_label.items():
                #     if i != j:
                    j_com = self.graph.node_label[j]
                    delta_modularity_2 = self.graph.tasks.compute_delta_modularity(j_com, i)
                    candidate_memberships[self.graph.tasks.get_node_label(j)] = delta_modularity_1 + abs(
                        delta_modularity_2)
                max_com = max(candidate_memberships, key=candidate_memberships.get)
                largest_delta_modularity_value = candidate_memberships[max_com]
                if largest_delta_modularity_value > 0:
                    node_label_changes[i] = max_com
                if largest_delta_modularity_value < 0:
                    node_label_changes[i] = int(max(list(self.graph.node_label_profile.keys())) + 1)
        components = []
        node_label_profile = defaultdict(list)
        for key, val in sorted(node_label_changes.items()):
            node_label_profile[val].append(key)
        node_label_profile = dict(node_label_profile)
        for k, v in node_label_profile.items():
            components.append(v)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # Barber, M. J. (2007). Modularity and community detection in bipartite networks. Physical Review E, 76(6), 066102.
    def akef_modified_brim_spectral_bipartite_cluster_detection(self):
        edges = []
        tokens = []
        records = []
        for n in self.graph.nodes.keys():
            extracted_tokens = self.graph.node_attribute[n]
            records.append(n)
            for t in extracted_tokens:
                edges.append((t, n))
                tokens.append(t)
        tokens = list(set(tokens))
        all_nodes = records + tokens
        edges = list(set(edges))
        token_membership = []
        record_membership = []
        memberships = []
        for mem, n in enumerate(all_nodes):
            if n in tokens:
                token_membership.append(mem)
            if n in records:
                record_membership.append(mem)
            memberships.append(mem)
        # Dimensions of the matrix
        p = len(records)
        q = len(tokens)
        c = len(all_nodes)
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        rg = {tokens[i]: i for i in range(0, q)}
        gn = {records[i]: i for i in range(0, p)}
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
        T_ed = zip([gn[j] for j in [i for i in records]], record_membership)
        T0 = np.zeros((p, c))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = zip([rg[j] for j in [i for i in tokens]], token_membership)
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
        record_memberships = {}
        for tup in tar_membership:
            record_memberships[str(tup[0])] = int(tup[1])
        grouped_record_memberships = defaultdict(list)
        for index, row in sorted(record_memberships.items()):
            grouped_record_memberships[row].append(index)
        components = []
        for k, v in grouped_record_memberships.items():
            components.append(v)
        cluster_list = []
        for cc in components:
            least_node = min(cc)
            for n in cc:
                cluster_list.append((str(least_node), str(n)))
        return cluster_list

    # SPEK

    # Define nested function for removing stop words
    def __token_frequencies(self):
        tokenCnt = 0
        refCnt = 0
        numTokenCnt = 0
        tokenFreqDict = {}
        for key in self.graph.node_attribute:
            refCnt += 1
            tokenList = self.graph.node_attribute[key]
            tokenCnt = tokenCnt + len(tokenList)
            for j in range(0, len(tokenList)):
                token = tokenList[j]
                if token.isdigit():
                    numTokenCnt += 1
                if token in tokenFreqDict:
                    tokenFreqDict[token] += 1
                else:
                    tokenFreqDict[token] = 1
        return tokenFreqDict

    def __readjust_graph_scoring_matrix(self):
        new_edges = []
        for e in self.graph.edges:
            s_n = e[0]
            s_ref = self.graph.node_attribute[s_n]
            set_s_ref = set(s_ref)
            t_n = e[1]
            t_ref = self.graph.node_attribute[t_n]
            set_t_ref = set(t_ref)
            if set_s_ref == set_t_ref or set.issubset(set_s_ref, set_t_ref) or set.issubset(set_t_ref, set_s_ref):
                new_edges.append((str(s_n), str(t_n), 1.0))
            else:
                s_difference_t = self.__remove_stop_words(list(set_s_ref.difference(set_t_ref)))
                t_difference_s = self.__remove_stop_words(list(set_t_ref.difference(set_s_ref)))
                p = DWM65_ScoringMatrixStd.normalized_similarity(s_difference_t, t_difference_s)
                if p > DWM10_Parms.mu:
                    new_edges.append((str(s_n), str(t_n), p))
        return self.graph.recast_graph(new_edges)


    def __readjust_graph_hierarchical_document_digit_descrimiator(self):
        new_edges = []
        for e in self.graph.edges:
            s_n = e[0]
            s_ref = self.graph.node_attribute[s_n]
            t_n = e[1]
            t_ref = self.graph.node_attribute[t_n]
            s_ref = self.__remove_stop_words(s_ref)
            t_ref = self.__remove_stop_words(t_ref)
            p = DWM69_HierarchicalDocumentComparatorWithDigitDiscriminator.decision(s_ref, t_ref)
            if p != 0:
                new_edges.append((s_n,t_n,p))
        return self.graph.recast_graph(new_edges)


    def __readjust_graph_euclidean_distance(self):
        eigenvals, eigenvecs = scipy.linalg.eig(self.graph.laplacian_weighted.todense())
        # eigenvals, eigenvecs = scipy.sparse.linalg.eigs(subgraph_object.laplacian_weighted,k=int(len(subgraph_object.node_name)-2),which="SM")
        # eigenvecs = np.array(np.around(eigenvecs.real, 3))
        # eigenvals = np.array(np.around(eigenvals.real, 3))
        print(len(self.graph.node_name))
        print(eigenvals.shape)
        print(eigenvecs.shape)
        eigenvecs = np.array(eigenvecs.real)
        eigenvals = np.array(eigenvals.real)
        eigenvals_sorted = np.sort(eigenvals)
        eigenvecs_transposed = np.transpose(eigenvecs)
        fiedler_vectors = []
        for ev in range(1, len(eigenvals_sorted)):
            # scalar = 1
            # if eigenvals_sorted[v]>0:
            # scalar = 1 / np.sqrt(eigenvals_sorted[ev])
            # scaled_eigenvecs_normalized = eigenvecs_transposed[eigenvals.tolist().index(eigenvals_sorted[ev])] * eigenvals_sorted[ev]
            corresponding_eigenval_location = eigenvals.tolist().index(eigenvals_sorted[ev])
            corresponding_eigenval = eigenvals[corresponding_eigenval_location]
            scaled_eigenvecs = eigenvecs_transposed[corresponding_eigenval_location] * corresponding_eigenval
            # scaled_eigenvecs = eigenvecs_transposed[corresponding_eigenval_location]

            # scaled_eigenvecs = eigenvecs_transposed[eigenvals.tolist().index(eigenvals[ev])]
            # scaled_eigenvecs = eigenvecs_transposed[ev] * eigenvals[ev]
            # scaled_eigenvecs = eigenvecs_transposed[ev] * eigenvals[ev]
            # scaled_normalized_eigenvecs = list(np.array(scaled_eigenvecs) / np.linalg.norm(scaled_eigenvecs))
            # scaled_eigenvecs_normalized = eigenvecs_transposed[eigenvals.tolist().index(eigenvals_sorted[v])] * scalar
            # scaled_eigenvecs_normalized = eigenvecs_transposed[eigenvals.tolist().index(eigenvals_sorted[v])]
            # norm = np.linalg.norm(scaled_eigenvecs_normalized)
            # if norm!=0:
            #     fiedler_vectors.append(scaled_eigenvecs_normalized/norm)
            # else:
            fiedler_vectors.append(scaled_eigenvecs)

        X = np.array(np.matrix(fiedler_vectors).T)
        subgraph_size = 5
        number_of_subgraphs = int(np.ceil(len(self.graph.node_name) / subgraph_size))
        clustering = AgglomerativeClustering().fit(X)
        print(clustering.labels_)

        # # # figure out K
        # # subgraph_size = 5
        # # number_of_subgraphs = np.ceil(len(v) / subgraph_size)
        #
        # distances = []
        # equivalent_nodes = []
        # all_nodes = []
        # # new_edges = []
        # # cosine_distances = {}
        # for nri1, vec1 in enumerate(X):
        #     n1_name = str(self.graph.node_inverted_index[nri1])
        #     # print(n1_name)
        #
        #     for nri2, vec2 in enumerate(X):
        #         n2_name = str(self.graph.node_inverted_index[nri2])
        #         if n1_name < n2_name:
        #             # cd = abs(self.graph.tasks.cosine_distance(vec1, vec2))
        #             ed = DWM_GEP_GraphOG_Math.euclidean_distance(vec1, vec2)
        #
        #             distances.append(ed)
        #             all_nodes.append(n1_name)
        #             equivalent_nodes.append(n2_name)
        # normalized_distances = []
        # if len(distances) > 1:
        #     normalized_distances = list(np.around(np.array((np.array(distances) - np.min(np.array(distances))) / (
        #             np.max(np.array(distances)) - np.min(np.array(distances)))), 3))
        # else:
        #     if distances[0] > 0:
        #         normalized_distances.append(1.0)
        #
        # # normalized_distances = list(np.array(distances) / np.linalg.norm(distances))
        # mean_normalized_distance = np.mean(np.array(normalized_distances))
        # print(mean_normalized_distance)
        # node_edges_distances = list(zip(all_nodes, equivalent_nodes, normalized_distances))
        # # node_edges_distances = list(filter(lambda x: x[2]>=mean_normalized_distance, node_edges_distances))
        # # new_edges = new_edges + node_edges_distances
        # # new_edges = list(set(new_edges))
        # # new_edges.sort()
        # print(node_edges_distances)
        # node_edges_distances = []
        # node_edges_distances.append(("1","1",1.0))
        # return self.graph.recast_graph(node_edges_distances)



    def akef_spectral_partioning_using_entity_kernels_step_1(self):
        cluster_list = []
        for k, v in self.graph.node_label_profile.items():
            if len(v) > 1:
                subgraph_object = self.graph.subgraph(v)
                subgraph_object.info()
                readjust_graph_object = subgraph_object.algorithms.__readjust_graph_hierarchical_document_digit_descrimiator()
                readjust_graph_object.info()
                cluster_list = cluster_list + readjust_graph_object.algorithms.talburt_transitive_closure_connected_component_detection()
                print("------")
                # print("---")
                # cluster_list = cluster_list + recasted_graph_object.algorithms.networkx_louvain_community_detection()
            else:
                for n in v:
                    cluster_list.append((n,n,1.0))
        return cluster_list

    def akef_spectral_partioning_using_entity_kernels_step_2(self):
        cluster_list = []
        for k, v in self.graph.node_label_profile.items():
            if len(v) > 1:
                subgraph_object = self.graph.subgraph(v)
                subgraph_object.info()
                print(v)
                # subgraph_object.algorithms.__readjust_graph_euclidean_distance()
                # recasted_graph_object = subgraph_object.algorithms.__readjust_graph_euclidean_distance()
                # recasted_graph_object.info()
                # cluster_list = cluster_list + subgraph_object.algorithms.networkx_louvain_community_detection()
                # cluster_list = cluster_list + subgraph_object.algorithms.akef_greedy_modularity_optimization_community_detection()
                cluster_list = cluster_list + subgraph_object.algorithms.akef_modified_brim_spectral_bipartite_cluster_detection()
                # print("------")
                # cluster_list = cluster_list + recasted_graph_object.algorithms.networkx_louvain_community_detection()
                # cluster_list = cluster_list + recasted_graph_object.algorithms.talburt_transitive_closure_connected_component_detection()
            else:
                for n in v:
                    cluster_list.append((n,n,1.0))
        return cluster_list

