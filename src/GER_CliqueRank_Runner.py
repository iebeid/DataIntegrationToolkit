import random
import operator
import numpy as np


def get_record_node_neighborhood(record_node, target_node, record_graph):
    neighborhood = []
    for record_pair_id, record_graph_dict in record_graph.items():
        if record_graph_dict["source"] == record_node:
            neighborhood.append(record_graph_dict["target"])
        if record_graph_dict["target"] == record_node:
            neighborhood.append(record_graph_dict["source"])
    if target_node in neighborhood:
        neighborhood.remove(target_node)
    if record_node in neighborhood:
        neighborhood.remove(record_node)
    return neighborhood


def get_record_node_neighborhood_for_random_walk(record_node, record_graph):
    neighborhood = []
    for record_pair_id, record_graph_dict in record_graph.items():
        if record_graph_dict["source"] == record_node:
            neighborhood.append(record_graph_dict["target"])
        if record_graph_dict["target"] == record_node:
            neighborhood.append(record_graph_dict["source"])
    if record_node in neighborhood:
        neighborhood.remove(record_node)
    return neighborhood


def get_record_edge_weight(node1, node2, record_graph):
    s = 0
    for record_pair_id, record_graph_dict in record_graph.items():
        pair_node_list = [record_graph_dict["source"], record_graph_dict["target"]]
        if (node1 in pair_node_list) and (node2 in pair_node_list):
            s = record_graph_dict["weight"]
    return s


def get_sum_of_weights_for_neighborhood(node, alpha, record_graph, neighborhood_nodes):
    sum_s = 0
    for nn in neighborhood_nodes:
        s = get_record_edge_weight(node, nn, record_graph)
        sum_s = sum_s + (s ** alpha)
    return sum_s


# Try to find if both nodes exist in the graph. The first node could be in the source
def find_edge(node1, node2, record_graph):
    found_record = 0
    for record_pair_id, record_graph_dict in record_graph.items():
        pair_nodes_list = [record_graph_dict["source"], record_graph_dict["target"]]
        if (node1 in pair_nodes_list) and (node2 in pair_nodes_list):
            found_record = record_graph_dict
        else:
            found_record = 0
    return found_record


def get_pair_node_id(record_pair_id, record_graph):
    for index, record_graph_dict in record_graph.items():
        if index == record_pair_id:
            return record_graph_dict["source"], record_graph_dict["target"]


def random_walker(start_node, target_node, steps, alpha, record_graph):
    current_node = start_node
    next_node = current_node
    for s in range(0, steps):
        neighborhood_nodes = get_record_node_neighborhood(current_node, target_node, record_graph)
        neighborhood_nodes_for_random_walk = get_record_node_neighborhood_for_random_walk(current_node, record_graph)
        P_bs = {}
        b = random.uniform(0, 1)
        s_current_target = get_record_edge_weight(current_node, target_node, record_graph)
        s_dash_curent_target = ((1 + b) ** alpha) * (s_current_target ** alpha)
        sum_s_r_i_r_k = 0
        for neighbor_node_k in neighborhood_nodes:
            sum_s_r_i_r_k = sum_s_r_i_r_k + (
                float(get_record_edge_weight(current_node, neighbor_node_k, record_graph))) ** alpha
        norm_r_i_r_t = s_dash_curent_target + sum_s_r_i_r_k
        for neighbor_node_j in neighborhood_nodes_for_random_walk:
            b = random.uniform(0, 1)
            s_r_i_r_j = get_record_edge_weight(current_node, neighbor_node_j, record_graph)
            # Calculate P_b
            if neighbor_node_j == target_node:
                numerator_1 = ((1 + b) ** alpha) * (s_r_i_r_j ** alpha)
                P_b_1 = numerator_1 / norm_r_i_r_t
                P_bs[neighbor_node_j] = P_b_1
            else:
                numerator_2 = (s_r_i_r_j ** alpha)
                P_b_2 = numerator_2 / norm_r_i_r_t
                P_bs[neighbor_node_j] = P_b_2
        sorted_P_bs = dict(sorted(P_bs.items(), key=operator.itemgetter(1), reverse=True))
        next_node = next(iter(sorted_P_bs.keys()))
        found_edge = find_edge(next_node, target_node, record_graph)
        if next_node == target_node:
            return 1
        if found_edge == 0:
            return 0
        current_node = next_node
    return 0


def random_surfer(record_graph, number_of_random_walks, steps, alpha):
    # {"source":source, "target":target, "weight": weight_s}
    for record_pair_id, record_graph_dict in record_graph.items():
        r_i = record_graph_dict["source"]
        r_j = record_graph_dict["target"]
        c_1 = 0
        c_2 = 0
        for m in range(0, int(number_of_random_walks / 2)):
            c_1 = c_1 + random_walker(r_i, r_j, steps, alpha, record_graph)
        for m in range(0, int(number_of_random_walks / 2)):
            c_2 = c_2 + random_walker(r_j, r_i, steps, alpha, record_graph)
        p_i_j = float((c_1 + c_2) / number_of_random_walks)
        record_graph_dict["matching_prob"] = float(p_i_j)


def run_clique_rank(bipartite_graph, number_of_random_walks, steps, alpha):
    record_graph = bipartite_graph["record_graph_dict"]
    bipartite_edges = bipartite_graph["bipartite_edges_dict"]
    random_surfer(record_graph, number_of_random_walks, steps, alpha)
    # {"source": pair_node_id, "target": token_node_id, "weight": p}
    for be_index, be_row in bipartite_edges.items():
        # I need to find pair node id
        source_node, target_node = get_pair_node_id(be_row["source"], record_graph)
        if source_node and target_node:
            found_record_edge = find_edge(source_node, target_node, record_graph)
            if found_record_edge != 0:
                be_row["weight"] = found_record_edge["matching_prob"]
    return bipartite_graph


def run_clique_rank_power_iteration(bipartite_graph, record_index, number_of_records, number_of_random_walks, steps, alpha):
    record_graph = bipartite_graph["record_graph_dict"]
    M_t = np.zeros((number_of_records, number_of_records), dtype=float)
    for i, r_x in record_graph.items():
        source_index = record_index[r_x["source"]]
        target_index = record_index[r_x["target"]]
        M_t[source_index][target_index]
    return bipartite_graph
