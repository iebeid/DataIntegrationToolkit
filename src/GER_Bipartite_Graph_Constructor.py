import numpy as np
import random

def construct_bipartite_graph(data_block, randomly_initialize_matching_probabilities):
    # Create token nodes and initialize X_t
    all_tokens = []
    for r_id, r in data_block.items():
        record_tokens = r.split()
        all_tokens = all_tokens + record_tokens
    unique_tokens = np.unique(np.array(all_tokens))
    token_nodes = {}
    j = 0
    for t in unique_tokens:
        # x_t = random.uniform(0, 1)
        x_t = 1.0
        N_t = 0
        for r_id, r in data_block.items():
            record_tokens = r.split()
            if t in record_tokens:
                N_t = N_t + 1
        token_nodes["T"+str(j)] = {"index": str(j), "token": str(t), "number_of_occurances": int(N_t), "weight": float(x_t)}
        j = j + 1
    # Create record pair nodes
    record_pair_nodes = {}
    i = 0
    for r_1_id, r_1 in data_block.items():
        for r_2_id, r_2 in data_block.items():
            s = 0
            if r_1_id != r_2_id:
                if r_1_id < r_2_id:
                    record_pair_nodes["R"+str(i)] = {"index": str(i), "source": r_1_id, "target": r_2_id, "weight": s}
                if r_2_id < r_1_id:
                    record_pair_nodes["R" + str(i)] = {"index": str(i), "source": r_2_id, "target": r_1_id, "weight": s}
                i = i + 1
    # Create edges
    edges = {}
    to_be_deleted = []
    edge_id = 0
    for pair_node_id, record_pair_dict in record_pair_nodes.items():
        r1 = data_block[record_pair_dict["source"]].split()
        r2 = data_block[record_pair_dict["target"]].split()
        shared_tokens = 0
        for token_node_id, token_dict in token_nodes.items():
            if (token_dict["token"] in r1) and (token_dict["token"] in r2):
                p = 0
                if randomly_initialize_matching_probabilities:
                    p = random.uniform(0, 1)
                else:
                    p = 1.0
                shared_tokens = shared_tokens + 1
                edges[edge_id] = {"source": str(pair_node_id), "target": str(token_node_id), "weight": p}
                edge_id = edge_id + 1
        if shared_tokens == 0:
            to_be_deleted.append(pair_node_id)

    for ob in to_be_deleted:
        del record_pair_nodes[ob]

    return {"term_dict": token_nodes, "record_graph_dict": record_pair_nodes, "bipartite_edges_dict": edges}


def construct_bipartite_graph_power_iteration(data_block):
    # Create token nodes
    all_tokens = []
    for r_id, r in data_block.items():
        record_tokens = r.split()
        all_tokens = all_tokens + record_tokens
    unique_tokens = np.unique(np.array(all_tokens))
    token_nodes = {}
    j = 0
    size_of_term_nodes = len(unique_tokens)

    X = np.zeros(size_of_term_nodes, dtype=float)
    D = np.zeros((size_of_term_nodes, size_of_term_nodes), dtype=float)
    for t in unique_tokens:
        x_t = random.uniform(0, 1)
        N_t = 0
        for r_id, r in data_block.items():
            record_tokens = r.split()
            if t in record_tokens:
                N_t = N_t + 1
        token_nodes["T"+str(j)] = {"index": str(j), "token": str(t), "number_of_occurances": int(N_t), "weight": float(x_t)}

        P_t = (N_t * (N_t - 1)) / 2
        D[j][j] = float(P_t)
        X[j] = float(x_t)
        j = j + 1
    # Create record pair nodes
    record_pair_nodes = {}
    records = {}
    i = 0
    j = 0
    for r_1_id, r_1 in data_block.items():
        records[r_1_id] = {"index": str(j)}
        for r_2_id, r_2 in data_block.items():
            s = 0
            if r_1_id != r_2_id:
                if r_1_id < r_2_id:
                    record_pair_nodes["R" + str(i)] = {"index": str(i), "source": r_1_id, "target": r_2_id, "weight": s}
                if r_2_id < r_1_id:
                    record_pair_nodes["R" + str(i)] = {"index": str(i), "source": r_2_id, "target": r_1_id, "weight": s}
                i = i + 1
    # Create edges
    edges = {}
    to_be_deleted = []
    edge_id = 0
    size_of_pair_nodes = len(record_pair_nodes)
    S = np.zeros((size_of_term_nodes, size_of_pair_nodes), dtype=float)
    C = np.zeros((size_of_pair_nodes, size_of_pair_nodes), dtype=float)
    Y = np.zeros(size_of_pair_nodes, dtype=float)


    for pair_node_id, record_pair_dict in record_pair_nodes.items():
        r1 = data_block[record_pair_dict["source"]].split()
        r2 = data_block[record_pair_dict["target"]].split()
        shared_tokens = 0
        p = random.uniform(0, 1)
        i = int(record_pair_dict["index"])
        C[i][i] = p
        for token_node_id, token_dict in token_nodes.items():
            if (token_dict["token"] in r1) and (token_dict["token"] in r2):
                shared_tokens = shared_tokens + 1
                edges[edge_id] = {"source": str(pair_node_id), "target": str(token_node_id), "weight": p}
                edge_id = edge_id + 1
                S[int(token_dict["index"])][int(record_pair_dict["index"])]=1.0
        if shared_tokens == 0:
            to_be_deleted.append(pair_node_id)

    for ob in to_be_deleted:
        del record_pair_nodes[ob]

    return {"term_dict": token_nodes, "record_graph_dict": record_pair_nodes, "bipartite_edges_dict": edges, "S":S, "C": C, "D": D, "X": X, "Y": Y}