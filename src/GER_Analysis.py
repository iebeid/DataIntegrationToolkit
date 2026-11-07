import pickle
import pandas as pd

import GER_Clusterer
import GER_Evaluater
# import GER_Visualizer


def main():
    sample_file = "data/S2G.txt"
    ground_truth_file = "evaluation/truthABCgoodDQ.txt"
    # ground_truth_file = "evaluation/truthABCpoorDQ.txt"
    # ground_truth_file = "evaluation/truthGeCo.txt"
    # ground_truth_file = "evaluation/truthRestaurant.txt"

    # final_matches_file = "objects/final_matches-s1-next.npy"
    # final_matches_file = "objects/final_matches-s2.npy"
    # final_matches_file = "objects/final_matches-s3.npy"
    final_matches_file = "objects/final_matches-s4.npy"
    # final_matches_file = "objects/final_matches-s5.npy"
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
    sample_dataframe = pd.read_csv(sample_file, sep=",", encoding="utf-8")
    N = len(sample_dataframe.index)
    # Load final matches
    file = open(final_matches_file, "rb")
    final_matches = pickle.load(file)
    file.close()
    print("Clustering")
    cluster_list = GER_Clusterer.cluster(final_matches)
    for j in cluster_list:
        print(j)
    print("Evaluating")
    GER_Evaluater.evaluate_clusters(cluster_list, N, ground_truth_file)
    # print("Visualizing")
    # GER_Visualizer.visualize(final_matches, cluster_list)


if __name__ == "__main__":
    main()