import pandas as pd


def main():
    print("Start Data Loading ...")
    # Load datasets
    # Find unique clusters in the merged truth set file
    # Draw random clusters from the merged truth set file
    # Merge the 200k and the 300k truth sets
    # For each unique truth set cluster group by records
    # Populate the drawn references
    # Measure error for each drawn sample
    list_A_200_k_df = pd.read_csv("data/ListA_200k.txt", sep="\t", encoding='utf-8')
    list_B_200_k_df = pd.read_csv("data/ListB_200k.txt", sep=",", encoding='utf-8')
    list_C_200_k_df = pd.read_csv("data/ListC_200k.txt", sep="|", encoding='utf-8')
    list_A_300_k_df = pd.read_csv("data/ListA_300k.txt", sep="\t", encoding='utf-8')
    list_B_300_k_df = pd.read_csv("data/ListB_300k.txt", sep=",", encoding='utf-8')
    list_C_300_k_df = pd.read_csv("data/ListC_300k.txt", sep="|", encoding='utf-8')
    print("1")
    print("data/ListA_200k.txt")
    print(list_A_200_k_df.info())
    print("--------------------")
    print("2")
    print("data/ListB_200k.txt")
    print(list_B_200_k_df.info())
    print("--------------------")
    print("3")
    print("data/ListC_200k.txt")
    print(list_C_200_k_df.info())
    print("--------------------")
    print("4")
    print("data/ListA_300k.txt")
    print(list_A_300_k_df.info())
    print("--------------------")
    print("5")
    print("data/ListB_300k.txt")
    print(list_B_300_k_df.info())
    print("--------------------")
    print("6")
    print("data/ListC_300k.txt")
    print(list_C_300_k_df.info())
    print("--------------------")
    print("List A merged")
    list_A_df = pd.concat([list_A_200_k_df, list_A_300_k_df])
    list_A_df = list_A_df.reset_index()
    print(list_A_df.info())
    print("--------------------")
    print("List B merged")
    list_B_df = pd.concat([list_B_200_k_df, list_B_300_k_df])
    list_B_df = list_B_df.reset_index()
    print(list_B_df.info())
    print("--------------------")
    print("List C merged")
    list_C_df = pd.concat([list_C_200_k_df, list_C_300_k_df])
    list_C_df = list_C_df.reset_index()
    print(list_C_df.info())
    print("--------------------")
    print("Load truth set ...")
    truth_200_k_df = pd.read_csv("data/truthABC_200k.txt", sep=",", encoding='utf-8')
    truth_300_k_df = pd.read_csv("data/truthABC_300k.txt", sep=",", encoding='utf-8')
    truth_200_k_df["idtruth"] = truth_200_k_df["idtruth"] * 10
    print("Merge truth set ...")
    merged_truth_set_df = pd.concat([truth_200_k_df,truth_300_k_df])
    merged_truth_set_df = merged_truth_set_df.reset_index()
    print(merged_truth_set_df.info())
    print("--------------------")
    unique_truth_clusters = list(merged_truth_set_df["idtruth"].unique())
    print("Number of clusters in truth set: " + str(len(unique_truth_clusters)))
    print("Cluster profile in truth set ...")
    print(merged_truth_set_df.groupby(['idtruth']).count())



    # Error scoring calculation. Count number of times each recid appeared in error lookup.
    with open('data/full_error_lookup.txt') as f:
        lines = f.readlines()
        for uc in unique_truth_clusters:
            cluster_records_df = merged_truth_set_df[merged_truth_set_df["idtruth"] == uc]
            list_of_rec_id = list(cluster_records_df["recid"].unique())
            for recid in list_of_rec_id:
                counter = 0
                for line in lines:
                    tokenized_line = [x.strip() for x in line.split(",")]
                    if recid in tokenized_line:
                        counter = counter + 1
                print("REC_ID: " + str(recid))
                print("Counter: " + str(counter))

    









    


if __name__ == '__main__':
    main()