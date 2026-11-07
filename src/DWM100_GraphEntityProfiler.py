import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import seaborn as sns
import networkx as nx
import modules.DWM10_Parms as DWM10_Parms
import modules.DWM67_JaccardComparator as DWM67_JaccardComparator
from modules.GraphEntityProfiler import DWM_GEP_GraphOG
sns.set()


def run_modularity_embedding_model(g):
    step_2_minimum_component_size = 2
    step_3_minimum_component_size = 2
    # -----------------------------
    # Step 1: Find connected components
    print("-------------- STEP 1 INITIATED --------------")
    print("-------------- STEP 1 INITIATED --------------", file=DWM10_Parms.logFile)
    # cluster_list = g.algorithms.talburt_transitive_closure_connected_component_detection()
    # g.update_node_labels(cluster_list)
    cluster_list = g.algorithms.akef_breadth_first_connected_component_detection()
    g.update_node_labels(cluster_list)
    # g.algorithms.networkx_breadth_first_connected_component_detection()
    # g.algorithms.scipy_depth_first_connected_component_detection()
    print("-------------- STEP 1 COMPLETED --------------")
    print("-------------- STEP 1 COMPLETED --------------", file=DWM10_Parms.logFile)
    # ------------------------------
    # Step 2: breakdown connected components using louvain or greedy modularity optimization
    print("-------------- STEP 2 INITIATED --------------")
    print("-------------- STEP 2 INITIATED --------------", file=DWM10_Parms.logFile)
    entity_clusters = []
    for component in g.node_label_profile.values():
        if len(component) >= step_2_minimum_component_size:
            sg = g.subgraph(component,True)
            sg.info()
            initial_modularity = sg.tasks.compute_modularity()
            cluster_list = sg.algorithms.akef_hierarchical_modularity_graph_embedding_for_record_clustering()
            sg.update_node_labels(cluster_list)
            final_modularity = sg.tasks.compute_modularity()
            difference_in_modularity = final_modularity - initial_modularity
            if difference_in_modularity == 0:
                cluster_list = []
                least_node = min(component)
                for n in component:
                    cluster_list.append((least_node,n))
            entity_clusters.extend(cluster_list)
        else:
            least_node = min(component)
            for n in component:
                entity_clusters.append((str(least_node), str(n)))
    g.update_node_labels(entity_clusters)
    print("-------------- STEP 2 COMPLETED --------------")
    print("-------------- STEP 2 COMPLETED --------------", file=DWM10_Parms.logFile)

def run_modularity_optimizaton_model(g):
    step_2_minimum_component_size = 2
    step_3_minimum_component_size = 2
    # -----------------------------
    # Step 1: Find connected components
    print("-------------- STEP 1 INITIATED --------------")
    print("-------------- STEP 1 INITIATED --------------", file=DWM10_Parms.logFile)
    cluster_list = g.algorithms.talburt_transitive_closure_connected_component_detection()
    g.update_node_labels(cluster_list)
    # g.algorithms.akef_breadth_first_connected_component_detection()
    # g.algorithms.networkx_breadth_first_connected_component_detection()
    # g.algorithms.scipy_depth_first_connected_component_detection()
    print("-------------- STEP 1 COMPLETED --------------")
    print("-------------- STEP 1 COMPLETED --------------", file=DWM10_Parms.logFile)
    # ------------------------------
    # Step 2: breakdown connected components using louvain or greedy modularity optimization
    print("-------------- STEP 2 INITIATED --------------")
    print("-------------- STEP 2 INITIATED --------------", file=DWM10_Parms.logFile)
    entity_clusters = []
    for component in g.node_label_profile.values():
        if len(component) >= step_2_minimum_component_size:
            sg = g.subgraph(component,True)
            sg.info()
            initial_modularity = sg.tasks.compute_modularity()
            # cluster_list = sg.algorithms.networkx_louvain_community_detection()
            cluster_list = sg.algorithms.akef_greedy_modularity_optimization_community_detection()
            sg.update_node_labels(cluster_list)
            final_modularity = sg.tasks.compute_modularity()
            difference_in_modularity = final_modularity - initial_modularity
            if difference_in_modularity == 0:
                cluster_list = []
                least_node = min(component)
                for n in component:
                    cluster_list.append((least_node,n))
            entity_clusters.extend(cluster_list)
        else:
            least_node = min(component)
            for n in component:
                entity_clusters.append((str(least_node), str(n)))
    g.update_node_labels(entity_clusters)
    print("-------------- STEP 2 COMPLETED --------------")
    print("-------------- STEP 2 COMPLETED --------------", file=DWM10_Parms.logFile)
    # ------------------------------
    # Step 3: bipartite spectral modularity clustering
    print("-------------- STEP 3 INITIATED --------------")
    print("-------------- STEP 3 INITIATED --------------", file=DWM10_Parms.logFile)
    entity_clusters = []
    for component in g.node_label_profile.values():
        if len(component) >= step_3_minimum_component_size:
            sg = g.subgraph(component, True)
            initial_modularity = sg.tasks.compute_modularity()
            cluster_list = sg.algorithms.akef_modified_brim_spectral_bipartite_cluster_detection()
            sg.update_node_labels(cluster_list)
            final_modularity = sg.tasks.compute_modularity()
            difference_in_modularity = final_modularity - initial_modularity
            if difference_in_modularity == 0:
                cluster_list = []
                least_node = min(component)
                for n in component:
                    cluster_list.append((least_node,n))
            entity_clusters.extend(cluster_list)
        else:
            least_node = min(component)
            for n in component:
                entity_clusters.append((str(least_node), str(n)))
    g.update_node_labels(entity_clusters)
    print("-------------- STEP 3 COMPLETED --------------")
    print("-------------- STEP 3 COMPLETED --------------", file=DWM10_Parms.logFile)
    return g


def run_spek_model(references, pair_list_object):
    step_2_minimum_component_size = 2
    # construct node label dict randomly assigned
    initial_nodes = {}
    l = 0
    for n, v in references.items():
        initial_nodes[str(n)] = {"type": "r", "attribute": list(v), "label": int(l)}
        l = l + 1
    initial_edges = pair_list_object
    g = DWM_GEP_GraphOG.GraphOG(initial_edges, nodes=initial_nodes, undirected=True, weighted=True)
    g.info()
    # -----------------------------
    # Step 1: Find connected components
    print("-------------- STEP 1 INITIATED --------------")
    print("-------------- STEP 1 INITIATED --------------", file=DWM10_Parms.logFile)
    g.algorithms.step_1_talburt_transitive_closure_connected_component_detection()
    # g.algorithms.step_1_akef_spectral_connected_component_detection()
    print("-------------- STEP 1 COMPLETED --------------")
    print("-------------- STEP 1 COMPLETED --------------", file=DWM10_Parms.logFile)
    # -----------------------------
    # Step 2: breakdown connected components using louvain or greedy modularity optimization
    print("-------------- STEP 2 INITIATED --------------")
    print("-------------- STEP 2 INITIATED --------------", file=DWM10_Parms.logFile)
    entity_clusters = []
    for component in g.node_label_profile.values():
        if len(component) >= step_2_minimum_component_size:
            tokens = []
            for n in component:
                att = g.node_attribute[n]
                for t in att:
                    tokens.append(t)
            tokens = list(set(tokens))
            nodes = {}
            l = 0, 0
            for n in component:
                att = g.node_attribute[n]
                nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
                l = l + 1
            for t in tokens:
                nodes[str(t)] = {"type": "t", "attribute": "empty", "label": int(l)}
                l = l + 1
            edges = []
            for t in tokens:
                for n, r in nodes.items():
                    token_similarity_scores = []
                    for t2 in r["attribute"]:
                        similarity_score = DWM67_JaccardComparator.normalized_similarity(t, t2)
                        token_similarity_scores.append(similarity_score)
                    max_sim_score = float(max(token_similarity_scores))
                    if max_sim_score > DWM10_Parms.mu:
                        edges.append((n, t, max_sim_score))
            edges.extend(pair_list_object)
            sg = DWM_GEP_GraphOG.GraphOG(edges, nodes=nodes, undirected=True, weighted=True)
            sg.info()
            sg.algorithms.akef_spectral_partioning_using_entity_kernels()
            if len(sg.node_label_profile.keys()) > 0:
                for n, c in sg.node_label_profile.items():
                    for t in c[:]:
                        if sg.node_type[t] == "t":
                            c.remove(t)
                    if len(c) != 0:
                        least_node = min(c)
                        for r in c:
                            entity_clusters.append((str(least_node), str(r)))
        else:
            least_node = min(component)
            for n in component:
                entity_clusters.append((str(least_node), str(n)))
    g.update_node_labels(entity_clusters)
    print("-------------- STEP 2 COMPLETED --------------")
    print("-------------- STEP 2 COMPLETED --------------", file=DWM10_Parms.logFile)
    # ------------------------------
    return g


def run_spek_model_2(references, pair_list_object):
    # create graph of tokens, records and entities
    tokens = []
    for n, att in references.items():
        for t in att:
            tokens.append(t)
    tokens = list(set(tokens))
    record_nodes = {}
    nodes = {}
    l = 0
    for n, att in references.items():
        nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
        record_nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
        l = l + 1
    for t in tokens:
        nodes[str(t)] = {"type": "t", "attribute": "empty", "label": int(l)}
        l = l + 1
    edges = []
    for t in tokens:
        for n, r in nodes.items():
            token_similarity_scores = []
            for t2 in r["attribute"]:
                similarity_score = DWM67_JaccardComparator.normalized_similarity(t, t2)
                # similarity_score = DWM65_ScoringMatrixStd.normalized_similarity(t, t2)
                token_similarity_scores.append(similarity_score)
            max_sim_score = float(max(token_similarity_scores))
            if max_sim_score >= DWM10_Parms.mu:
                edges.append((n, t, max_sim_score))
    edges.extend(pair_list_object)
    sg = DWM_GEP_GraphOG.GraphOG(edges, nodes=nodes, undirected=True, weighted=True)
    sg.info()
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_enriched.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_weighted.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_weighted_enriched.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_degree_weighted.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_enriched_degree_weighted.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_weighted_degree_weighted.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_weighted_enriched_degree_weighted.todense()) #nothing
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_degree_normalized.todense()) #good
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_enriched_degree_normalized.todense()) #good
    # eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_weighted_degree_normalized.todense()) #good
    eigenvals, eigenvecs = scipy.linalg.eig(sg.laplacian_weighted_enriched_degree_normalized.todense())  # good
    eigenvecs = eigenvecs.real
    eigenvals = eigenvals.real
    print(eigenvecs.shape)
    print(eigenvals.shape)
    print(eigenvals)
    d = 3
    first_eig_position = 1
    fiedler_vectors = []
    for i in range(first_eig_position, d):
        fiedler_pos_1 = np.where(eigenvals == np.sort(eigenvals)[i])[0][0]
        # n1 = (1 / np.sqrt(abs(eigenvals[fiedler_pos_1])))
        # n1 = (eigenvals[fiedler_pos_1] / np.sqrt(abs(eigenvals[fiedler_pos_1])))
        n1 = eigenvals[fiedler_pos_1]
        # n1 = 1.0
        fiedler_vector_1 = list(np.real(np.transpose(eigenvecs)[fiedler_pos_1]) * n1)
        fiedler_vectors.append(fiedler_vector_1)
    fiedler_vectors = np.matrix(fiedler_vectors)
    X = fiedler_vectors.T
    X_new = np.zeros((len(references.keys()), d - first_eig_position))
    for node_record, tokens in references.items():
        node_record_index = sg.node_index[node_record]
        neighors = sg.get_neighbors(node_record)
        neighbor_embedding = []
        neighbor_embedding.append(X[node_record_index])
        for n in neighors:
            n_index = sg.node_index[n]
            neighbor_embedding.append(X[n_index])
        neighbor_embedding_average = np.array(neighbor_embedding).sum(axis=0) / (len(neighors) + 1)
        X_new[node_record_index] = neighbor_embedding_average
    new_edges = []
    print(X_new)
    # trans = TSNE(n_components=2)
    # viz = trans.fit_transform(X_new)
    for nri1, vec1 in enumerate(X_new):
        n1_name = str(sg.node_inverted_index[nri1])
        for nri2, vec2 in enumerate(X_new):
            n2_name = str(sg.node_inverted_index[nri2])
            if n1_name < n2_name:
                cd = abs(sg.tasks.cosine_distance(vec1, vec2))
                print(cd)
                decision = "Not Match"
                if cd == 1.0:
                    decision = "Match"
                    new_edges.append((n1_name, n2_name, cd))
                # print("Pair: " + str(sg.node_inverted_index[nri1]) + " , " + str(sg.node_inverted_index[nri2]) + " --> " + str(decision))
    final_graph = DWM_GEP_GraphOG.GraphOG(new_edges, nodes=record_nodes, undirected=True, weighted=True)
    final_graph.info()
    final_graph.algorithms.step_1_akef_breadth_first_connected_component_detection()
    return final_graph


def run_spek_model_3(references, pair_list_object):
    # construct node label dict randomly assigned
    initial_nodes = {}
    l = 0
    for n, v in references.items():
        initial_nodes[str(n)] = {"type": "r", "attribute": list(v), "label": int(l)}
        l = l + 1
    initial_edges = pair_list_object
    g = DWM_GEP_GraphOG.GraphOG(initial_edges, nodes=initial_nodes, undirected=True, weighted=True)
    g.info()
    # -----------------------------
    # Step 1: Find connected components
    print("-------------- STEP 1 INITIATED --------------")
    print("-------------- STEP 1 INITIATED --------------", file=DWM10_Parms.logFile)
    g.algorithms.step_1_talburt_transitive_closure_connected_component_detection()
    # g.algorithms.step_1_akef_spectral_connected_component_detection_2()
    # g.algorithms.step_1_akef_spectral_connected_component_detection()
    print("-------------- STEP 1 COMPLETED --------------")
    print("-------------- STEP 1 COMPLETED --------------", file=DWM10_Parms.logFile)
    # ------------------------------
    return g


def run_spek_model_trial_4(references, pair_list_object):
    step_2_minimum_component_size = 2
    initial_nodes = {}
    l = 0
    for n, v in references.items():
        initial_nodes[str(n)] = {"type": "r", "attribute": list(v), "label": int(l)}
        l = l + 1
    graph_object = DWM_GEP_GraphOG.GraphOG(pair_list_object, nodes=initial_nodes)
    graph_object.info()
    # -----------------------------
    # Step 1: Find connected components
    print("-------------- STEP 1 INITIATED --------------")
    print("-------------- STEP 1 INITIATED --------------", file=DWM10_Parms.logFile)
    graph_object.algorithms.step_1_akef_breadth_first_connected_component_detection()
    # graph_object.algorithms.step_1_akef_spectral_connected_component_detection_2()
    # graph_object.algorithms.step_1_akef_spectral_connected_component_detection()
    # graph_object.algorithms.step_1_akef_spectral_connected_component_detection_2()
    # graph_object.algorithms.step_1_akef_spectral_connected_component_detection_3()
    # graph_object.algorithms.step_1_akef_spectral_connected_component_detection_4()

    print("-------------- STEP 1 COMPLETED --------------")
    print("-------------- STEP 1 COMPLETED --------------", file=DWM10_Parms.logFile)
    # -----------------------------
    # Step 2: breakdown connected components using louvain or greedy modularity optimization
    print("-------------- STEP 2 INITIATED --------------")
    print("-------------- STEP 2 INITIATED --------------", file=DWM10_Parms.logFile)
    initial_modularity = graph_object.tasks.compute_modularity()
    entity_clusters = []
    for component in graph_object.node_label_profile.values():
        if len(component) >= step_2_minimum_component_size:
            print(len(component))
            tokens = []
            for n in component:
                att = graph_object.node_attribute[n]
                for t in att:
                    tokens.append(t)
            tokens = list(set(tokens))
            nodes = {}
            l = 0
            for n in component:
                att = graph_object.node_attribute[n]
                nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
                l = l + 1
            for t in tokens:
                nodes[str(t)] = {"type": "t", "attribute": "empty", "label": int(l)}
                l = l + 1
            edges = []
            for t in tokens:
                for n, r in nodes.items():
                    token_similarity_scores = []
                    for t2 in r["attribute"]:
                        similarity_score = DWM67_JaccardComparator.normalized_similarity(t, t2)
                        token_similarity_scores.append(similarity_score)
                    max_sim_score = float(max(token_similarity_scores))
                    if max_sim_score >= DWM10_Parms.mu:
                        edges.append((n, t, max_sim_score))
            edges.extend(pair_list_object)
            sg = DWM_GEP_GraphOG.GraphOG(edges, nodes=nodes, undirected=True, weighted=True)
            sg.info()
            new_nodes, new_edges = sg.algorithms.akef_spectral_partioning_using_entity_kernels()

            # if len(sg.node_label_profile.keys()) > 0:
            for n, c in sg.node_label_profile.items():
                component_copy = c[:]
                for t in c:
                    if sg.node_type[t] == "t":
                        component_copy.remove(t)
                if len(component_copy) != 0:
                    least_node = min(component_copy)
                    for r in component_copy:
                        entity_clusters.append((str(least_node), str(r)))
        else:
            least_node = min(component)
            for n in component:
                entity_clusters.append((str(least_node), str(n)))
    print(entity_clusters)
    graph_object.update_node_labels(entity_clusters)
    print("-------------- STEP 2 COMPLETED --------------")
    print("-------------- STEP 2 COMPLETED --------------", file=DWM10_Parms.logFile)
    # ------------------------------
    return graph_object


def spek_model(graph_object):
    # build graph
    graph_object.info()
    print("---TRANSITIVE CLOSURE / CONNECTED COMPONENT DETECTION---")
    # block_list = graph_object.algorithms.akef_breadth_first_connected_component_detection()
    # cluster_list = graph_object.algorithms.talburt_transitive_closure_connected_component_detection()
    components = graph_object.algorithms.akef_quick_connected_components_detection()
    print(components)
    cluster_list = []
    for cc in components:
        least_node = min(cc)
        for n in cc:
            cluster_list.append((graph_object.node_inverted_index[least_node], graph_object.node_inverted_index[n]))
    graph_object.update_node_labels(cluster_list)
    # print("---SPEK 1---")
    # cluster_list = graph_object.algorithms.akef_spectral_partioning_using_entity_kernels_step_1()
    # graph_object.update_node_labels(cluster_list)
    # print("---SPEK 2---")
    # cluster_list = graph_object.algorithms.akef_spectral_partioning_using_entity_kernels_step_2()
    # graph_object.update_node_labels(cluster_list)
    return graph_object
