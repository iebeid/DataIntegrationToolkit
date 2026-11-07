import numpy as np

def list_difference(list_1, list_2):
    return list(set(list_1) - set(list_2)) + list(set(list_2) - set(list_1))



def match(original_data_frame, bipartite_graph, final_matches, matching_probability_threshold):
    record_graph_nodes = bipartite_graph["record_graph_dict"]
    # {"source": pair_node_id, "target": token_node_id, "weight": p}
    for be_index, be_dict in record_graph_nodes.items():
        if be_dict["matching_prob"] > matching_probability_threshold:
            final_matches.append((be_dict["source"], be_dict["target"]))

    # Validate final matches

    first_row = []
    second_row = []
    for e in final_matches:
        first_row.append(e[0])
        second_row.append(e[1])
    unique_records = np.unique(np.array(first_row+second_row)).tolist()
    all_records = original_data_frame["RecID"].tolist()
    single_clusters_list = list_difference(all_records,unique_records)
    single_clusters = []
    for r in single_clusters_list:
        single_clusters.append((r,r))

    final_matches = final_matches + single_clusters
    return final_matches