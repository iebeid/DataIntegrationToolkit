import pandas as pd
from tqdm import tqdm
import random
import operator


# Jaccard similarity
def jaccard(s1,s2):
    a = set(s1)
    b = set(s2)
    if len(a) == 0 and len(b) == 0:
        return 0
    return float(len(a&b) / len(a|b))


#Lists all the overlapping ngrams in a string (similar to a sliding window)
def ngrams(seq, n):
    return [seq[i:i+n] for i in range(1+len(seq)-n)]


# sliding window
def window(fseq, window):
    for i in range(len(fseq) - window + 1):
        yield fseq[i:i+window]


def main():
    data_file = "S9-By-ClusterID-Then-TrueID.txt"
    corpus_df_main = pd.read_csv(data_file, sep=",", encoding='utf-8')
    length_of_dataframe = len(corpus_df_main.index)


    C = {}
    for i_a, s_a in tqdm(corpus_df_main.iterrows()):
        C[str(s_a[0])] = str(s_a[3])
    # Parameters
    number_of_grams = 2
    ngram_sliding_window_size = 8
    blocking_factor = 20
    jaccard_threshold = 0.2
    #----------
    blocks = {}
    block_number = 0
    for b in tqdm(range(blocking_factor)):
        block_size = int(length_of_dataframe/blocking_factor)
        block = {}
        record_i_id, record_i = random.choice(list(C.items()))
        block[record_i_id] = record_i
        for j in range(block_size):
            record_k = ""
            n1 = ngrams(record_i.lower(), number_of_grams)
            s1 = {''.join(x) for x in window(n1, ngram_sliding_window_size)}
            jaccard_indices = {}
            for key, value in C.items():
                if str(key) != str(record_i_id):
                    n2 = ngrams(value.lower(), number_of_grams)
                    s2 = {''.join(x) for x in window(n2, ngram_sliding_window_size)}
                    jaccard_index = jaccard(s1,s2)
                    # if jaccard_index > jaccard_threshold:
                    jaccard_indices[key] = jaccard_index
            if jaccard_indices:
                sorted_jaccard_indices = dict(sorted(jaccard_indices.items(), key=operator.itemgetter(1), reverse=True))
                # print(sorted_jaccard_indices)
                record_k_id = list(sorted_jaccard_indices.keys())[0]
                record_k = C[record_k_id]
                block[record_k_id] = record_k
                del C[record_k_id]
                record_i = record_k
            else:
                continue
        block_number = block_number + 1
        blocks[block_number] = block

    for i,b in blocks.items():
        print("Block : " + str(i))
        print("----------------------------------")
        for r_id,r in b.items():
            print(str(r_id) + " : " + str(r))
        print("----------------------------------")


if __name__ == '__main__':
    main()