
import random
import networkx as nx
import igraph as ig
import community as louvain
import numpy as np
from collections import defaultdict



def networkx_modularity_graph_clustering(nodes, g):
    initial_networkx_modularity = louvain.modularity(nodes, g)
    print("Initial cluster modularity: " + str(initial_networkx_modularity))
    final_clusters = louvain.best_partition(g, partition=nodes)
    final_networkx_modularity = louvain.modularity(nodes, g)
    print("Final cluster modularity: " + str(final_networkx_modularity))
    return final_clusters

def cluster(final_matches):
    # #Informap
    # G = nx.Graph()
    # G.add_edges_from(final_matches)
    # g_ig = ig.Graph.Adjacency(
    #     (nx.to_numpy_matrix(G) > 0).tolist(), mode=ig.ADJ_UNDIRECTED
    # )
    # random.seed(123)
    # c_infomap = g_ig.community_infomap()
    # cluster_list = []
    # for clusters in c_infomap:
    #     for node in clusters:
    #         cluster_list.append((str(list(G.nodes)[int(clusters[0])]), str(list(G.nodes)[int(node)])))
    # prepare edges
    all_nodes = []
    for e in final_matches:
        sn = e[0]
        tn = e[1]
        all_nodes.append(sn)
        all_nodes.append(tn)
    unique_nodes = np.unique(np.array(all_nodes))
    nodes = {}
    com = 0
    for n in unique_nodes:
        nodes[n] = com
        com = com + 1
    g = nx.Graph()
    g.add_edges_from(final_matches)
    final_clusters = networkx_modularity_graph_clustering(nodes, g)

    grouped_final_clusters = defaultdict(list)
    for key, val in sorted(final_clusters.items()):
        grouped_final_clusters[val].append(key)

    # printing result
    print("Grouped dictionary is : " + str(dict(grouped_final_clusters)))

    cluster_list = []
    for cluster_id, records in dict(grouped_final_clusters).items():
        print(str(cluster_id) + " , " + str(records))
        least_record_value = min(list(records))
        for r in records:
            cluster_list.append((least_record_value, r))
    return cluster_list