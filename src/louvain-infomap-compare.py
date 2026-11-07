import random

import community as community_louvain
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import stellargraph as sg


def main():
    source_nodes = []
    target_nodes = []
    with open('karate_edges_77.txt') as fp:
        for e in fp:
            source_nodes.append(e[0])
            target_nodes.append(e[1])
    # Cluster using Louvain
    edges_df = pd.DataFrame({"source": source_nodes, "target": target_nodes})
    graph = sg.StellarGraph(edges=edges_df)
    # print(graph.info())
    G = graph.to_networkx()
    print(nx.info(G))
    # connected components analysis
    cc = nx.number_connected_components(G)
    print("Number of connected components: " + str(cc))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    cc_sizes = []
    cc_index = 0
    cc_results = []
    for cci in Gcc:
        cc_index = cc_index + 1
        for item in cci:
            print(item)
            cc_results.append((item, cc_index))
        cc_sizes.append(len(cci))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    langs = list(range(0, len(cc_sizes)))
    students = cc_sizes
    ax.bar(langs, students)
    plt.show()
    # compute the best partition using the community detection algorithm Louvain
    # https://github.com/taynaud/python-louvain
    clusters = community_louvain.best_partition(G)
    cluster_values = []
    for k, v in clusters.items():
        cluster_values.append(v)
    unique_clusters = np.unique(np.array(cluster_values)).tolist()
    print("Number of Louvain communities: " + str(len(unique_clusters)))
    # Infomap communities
    # translate the object into igraph
    g_ig = ig.Graph.Adjacency(
        (nx.to_numpy_matrix(G) > 0).tolist(), mode=ig.ADJ_UNDIRECTED
    )
    # convert via adjacency matrix
    print(g_ig.summary())
    # perform community detection
    random.seed(123)
    c_infomap = g_ig.community_infomap()
    print("Number of Infomap communities: ")
    print(c_infomap.summary())
    print(c_infomap)
    # plot the community sizes
    infomap_sizes = c_infomap.sizes()
    plt.title("Infomap community sizes")
    plt.xlabel("community id")
    plt.ylabel("number of nodes")
    plt.bar(list(range(1, len(infomap_sizes) + 1)), infomap_sizes)
    plt.show()
    # Modularity metric for infomap
    print(c_infomap.modularity)
    # assign community membership results back to networkx, keep the dictionary for later comparisons with the clustering
    infomap_com_dict = dict(zip(list(G.nodes()), c_infomap.membership))
    nx.set_node_attributes(G, infomap_com_dict, "c_infomap")
    # extraction of a subgraph from the nodes in this community
    com_id = 1
    com_G = G.subgraph(
        [n for n, attrdict in G.nodes.items() if attrdict["c_infomap"] == com_id]
    )
    print(nx.info(com_G))
    # plot community structure only
    pos = nx.random_layout(com_G, seed=123)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(com_G, pos, edge_color="#26282b", node_color="blue", alpha=0.3)
    plt.axis("off")
    plt.show()
    # Add block id to dataframe


if __name__ == '__main__':
    main()
