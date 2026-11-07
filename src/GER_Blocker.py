import random
import operator
import pandas as pd
from tqdm import tqdm
import GER_Tokenizer

# Jaccard similarity
def jaccard(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 0
    return float(len(s1&s2) / len(s1|s2))


#Lists all the overlapping ngrams in a string (similar to a sliding window)
def ngrams(seq, n):
    return [seq[i:i+n] for i in range(1+len(seq)-n)]


#Sliding window
def window(fseq, window):
    for i in range(len(fseq) - window + 1):
        yield fseq[i:i+window]


def block_jaccard(data_dict, number_of_blocks, ngram_sliding_window_size, number_of_grams):
    C = {}
    for s_a in tqdm(data_dict):
        C[str(s_a)] = str(data_dict[s_a])
    # Parameters
    # jaccard_threshold = 0.2 # In case we would like to restrict the jaccard similarity
    blocks = {}
    block_number = 0
    for b in tqdm(range(number_of_blocks)):
        block_size = int(len(data_dict)/number_of_blocks)
        block = {}
        record_i_id, record_i = random.sample(list(C.items()),1)[0]
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
                    # if jaccard_index > jaccard_threshold: # In case we would like to restrict the jaccard similarity
                    jaccard_indices[key] = jaccard_index
            if jaccard_indices:
                sorted_jaccard_indices = dict(sorted(jaccard_indices.items(), key=operator.itemgetter(1), reverse=True))
                record_k_id = list(sorted_jaccard_indices.keys())[0]
                record_k = C[record_k_id]
                block[record_k_id] = record_k
                del C[record_k_id]
                record_i = record_k
            else:
                continue
        block_number = block_number + 1
        blocks[block_number] = block
    return blocks