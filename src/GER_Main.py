import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np

import GER_Tokenizer
import GER_Blocker
import GER_Bipartite_Graph_Constructor
import GER_ITER_Runner
import GER_CliqueRank_Runner
import GER_Matcher


def main():
    # final_matches_file = "objects/final_matches-s1.npy"
    # final_matches_file = "objects/final_matches-s2.npy"
    # final_matches_file = "objects/final_matches-s3.npy"
    # final_matches_file = "objects/final_matches-s4.npy"
    final_matches_file = "objects/final_matches-s5.npy"
    # final_matches_file = "objects/final_matches-s6.npy"
    # final_matches_file = "objects/final_matches-s7.npy"
    # final_matches_file = "objects/final_matches-s8.npy"
    # final_matches_file = "objects/final_matches-s9.npy"
    # final_matches_file = "objects/final_matches-s10.npy"
    # final_matches_file = "objects/final_matches-s11.npy"
    # final_matches_file = "objects/final_matches-s12.npy"
    # final_matches_file = "objects/final_matches-s13.npy"
    # final_matches_file = "objects/final_matches-s14.npy"
    # final_matches_file = "objects/final_matches-s15.npy"
    # final_matches_file = "objects/final_matches-s16.npy"
    # final_matches_file = "objects/final_matches-s17.npy"
    # final_matches_file = "objects/final_matches-s18.npy"

    # sample_file = "data/S1G.txt"
    # sample_file = "data/S2G.txt"
    # sample_file = "data/S3Rest.txt"
    # sample_file = "data/S4G.txt"
    sample_file = "data/S5G.txt"
    # sample_file = "data/S6GeCo.txt"
    # sample_file = "data/S7GX.txt"
    # sample_file = "data/S8P.txt"
    # sample_file = "data/S9P.txt"
    # sample_file = "data/S10PX.txt"
    # sample_file = "data/S11PX.txt"
    # sample_file = "data/S12PX.txt"
    # sample_file = "data/S13GX.txt"
    # sample_file = "data/S14GX.txt"
    # sample_file = "data/S15GX.txt"
    # sample_file = "data/S16PX.txt"
    # sample_file = "data/S17PX.txt"
    # sample_file = "data/S18PX.txt"

    number_of_random_walks = 10
    steps = 20
    alpha = 20
    matching_probability_threshold = 0.98
    number_of_grams = 2
    ngram_sliding_window_size = 2
    number_of_blocks = 300
    epochs = 5
    convergence_threshold = 0.00001
    power_iter = False
    randomly_initialize_matching_probabilities = False

    sample_dataframe = pd.read_csv(sample_file, sep=",", encoding="utf-8")
    record_index = {}
    for index, row in sample_dataframe.iterrows():
        record_index[row["RecID"]] = index
    file_size = len(sample_dataframe.index)
    number_of_records_per_block = int(file_size/number_of_blocks)

    data_dict = GER_Tokenizer.tokenize(sample_dataframe)
    # data_dict = GER_Tokenizer.process_single_reference(sample_dataframe)
    data_blocks = GER_Blocker.block_jaccard(data_dict, number_of_blocks, ngram_sliding_window_size, number_of_grams)

    final_matches = []

    if not power_iter:
        # Controller
        for i,block in data_blocks.items():
            print("\nProcessing block : " + str(i))
            # Construct graph
            bipartite_graph = GER_Bipartite_Graph_Constructor.construct_bipartite_graph(block, randomly_initialize_matching_probabilities)
            # {"term_dict": token_nodes, "record_graph_dict": record_pair_nodes, "bipartite_edges_dict": edges}
            term_nodes = bipartite_graph["term_dict"]
            # {"index": int(j), "token": str(t), "number_of_occurances": int(N_t), "weight": float(x_t)}
            record_graph_nodes = bipartite_graph["record_graph_dict"]
            # {"index": int(i), "r1": r_1_id, "r2": r_2_id, "weight": s}
            bipartite_edges = bipartite_graph["bipartite_edges_dict"]
            # {"source": pair_node_id, "target": token_node_id, "weight": p}

            print("\n----------------------------------------")
            print("Number of unique tokens: " + str(len(term_nodes)))
            print("Number of pair nodes: " + str(len(record_graph_nodes)))
            print("Number of bipartite edges: " + str(len(bipartite_edges)))
            print("Number of record-record edges: " + str(len(record_graph_nodes)))
            print("----------------------------------------")
            for e in tqdm(range(epochs)):
                # ITER
                bipartite_graph = GER_ITER_Runner.run_iter(bipartite_graph)
                # CliqueRank
                bipartite_graph = GER_CliqueRank_Runner.run_clique_rank(bipartite_graph, number_of_random_walks, steps, alpha)

            # Matching
            print("\nMatching")
            final_matches = GER_Matcher.match(sample_dataframe, bipartite_graph, final_matches, matching_probability_threshold)

    if power_iter:
        # Controller Power Iteration Method
        for i,block in data_blocks.items():
            print("\nProcessing block : " + str(i))
            # Construct graph
            bipartite_graph = GER_Bipartite_Graph_Constructor.construct_bipartite_graph_power_iteration(block)
            # {"term_dict": token_nodes, "record_graph_dict": record_pair_nodes, "bipartite_edges_dict": edges, "S":S, "C": C, "D": D, "X": X, "Y": Y}
            term_nodes = bipartite_graph["term_dict"]
            # {"index": int(j), "token": str(t), "number_of_occurances": int(N_t), "weight": float(x_t)}
            record_graph_nodes = bipartite_graph["record_graph_dict"]
            # {"index": int(i), "r1": r_1_id, "r2": r_2_id, "weight": s}
            bipartite_edges = bipartite_graph["bipartite_edges_dict"]
            # {"source": pair_node_id, "target": token_node_id, "weight": p}
            S = bipartite_graph["S"]
            C = bipartite_graph["C"]
            D = bipartite_graph["D"]
            X = bipartite_graph["X"]
            Y = bipartite_graph["Y"]
            print("\n----------------------------------------")
            print("Number of unique tokens: " + str(len(term_nodes)))
            print("Number of pair nodes: " + str(len(record_graph_nodes)))
            print("Number of bipartite edges: " + str(len(bipartite_edges)))
            print("Number of record-record edges: " + str(len(record_graph_nodes)))
            print("----------------------------------------")
            converge = False
            while not converge:
                # ITER
                bipartite_graph, w1 = GER_ITER_Runner.run_iter_power_iteration(bipartite_graph)
                # CliqueRank
                bipartite_graph = GER_CliqueRank_Runner.run_clique_rank_power_iteration(bipartite_graph, record_index, number_of_records_per_block, number_of_random_walks, steps, alpha)

                # if np.sum(np.subtract(X, S.dot(C).dot(w1))) < convergence_threshold:
                #     converge = True

            # Matching
            print("\nMatching")
            final_matches = GER_Matcher.match(sample_dataframe, bipartite_graph, final_matches, matching_probability_threshold)

    #Save final matches
    fmf = open(final_matches_file, "wb")
    pickle.dump(final_matches, fmf)
    fmf.close()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()