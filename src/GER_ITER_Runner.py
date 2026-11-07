from itertools import groupby
from operator import itemgetter
import numpy as np


def run_iter(bipartite_graph):
    # {"term_dict": token_nodes, "record_graph_dict": record_pair_nodes, "bipartite_edges_dict": edges}
    term_nodes = bipartite_graph["term_dict"]
    # {"index": int(j), "token": str(t), "number_of_occurances": int(N_t), "weight": float(x_t)}
    record_graph_nodes = bipartite_graph["record_graph_dict"]
    # {"index": int(i), "r1": r_1_id, "r2": r_2_id, "weight": s}
    bipartite_edges = bipartite_graph["bipartite_edges_dict"]
    # {"source": pair_node_id, "target": token_node_id, "weight": p}
    edges_record_nodes = sorted(bipartite_edges.values(), key=itemgetter("source"))
    for key, value in groupby(edges_record_nodes, key=itemgetter("source")):
        for k1 in value:
            record_graph_nodes[key]["weight"] = record_graph_nodes[key]["weight"] + term_nodes[k1["target"]]["weight"]
    edges_term_nodes = sorted(bipartite_edges.values(), key=itemgetter("target"))
    for key, value in groupby(edges_term_nodes, key=itemgetter("target")):
        term_node_weight = term_nodes[key]["weight"]
        for k1 in value:
            p = k1["weight"]
            s = record_graph_nodes[k1["source"]]["weight"]
            N_t = term_nodes[key]["number_of_occurances"]
            P_t = (N_t * (N_t - 1)) / 2
            term_node_weight = term_node_weight + ((p * s) / P_t)
        term_node_weight = (term_node_weight / (term_node_weight + 1))
        term_nodes[key]["weight"] = term_node_weight
        term_node_weight = 0
    return bipartite_graph


def run_iter_power_iteration(bipartite_graph):
    bipartite_graph["Y"] = bipartite_graph["S"].transpose().dot(bipartite_graph["X"])
    w1 = np.array([])
    if np.linalg.det(bipartite_graph["D"]) != 0:
        bipartite_graph["X"] = np.linalg.inv(bipartite_graph["D"]).dot(bipartite_graph["S"]).dot(bipartite_graph["C"]).dot(bipartite_graph["Y"])
        # X = X / (X + 1)
        bipartite_graph["X"] = bipartite_graph["X"] / np.linalg.norm(bipartite_graph["X"])
        # Check for convergance
        w, v = np.linalg.eig(bipartite_graph["S"].transpose().dot(np.linalg.inv(bipartite_graph["D"])).dot(bipartite_graph["S"]).dot(bipartite_graph["C"]))
        v = np.array(v / np.linalg.norm(v))
        w1 = v.diagonal()
    else:
        bipartite_graph["X"] = bipartite_graph["D"].dot(bipartite_graph["S"]).dot(bipartite_graph["C"]).dot(bipartite_graph["Y"])
        # X = X / (X + 1)
        bipartite_graph["X"] = bipartite_graph["X"] / np.linalg.norm(bipartite_graph["X"])
        # Check for convergance
        w, v = np.linalg.eig(bipartite_graph["S"].transpose().dot(bipartite_graph["D"]).dot(bipartite_graph["S"]).dot(bipartite_graph["C"]))
        v = np.array(v / np.linalg.norm(v))
        w1 = v.diagonal()
    return bipartite_graph, w1
