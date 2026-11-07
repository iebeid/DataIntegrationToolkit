import numpy as np
from tqdm import tqdm


class FiltER():

    def __init__(self, encoding, beta, token_counts, corpus):
        print("----BLOCKING AND FILTERING----")
        self.beta = beta
        self.token_counts = token_counts
        self.corpus = corpus
        self.Y = {}
        self.L = []
        self.T = []
        self.blocks = []
        self.encoding = encoding

    def __compute_blocking_tokens(self):
        for id, document in self.corpus.items():
            for t in document:
                if self.token_counts[t] <= self.beta:
                    self.L.append((t, id))
        self.L.sort()

    def __get_blocking_tokens(self):
        for p in self.L:
            self.T.append(p[0])
        self.T = list(np.unique(np.array(self.T)))

    def __compute_blocks(self):
        for b_t in tqdm(self.T):
            B = []
            for p in self.L:
                if p[0] == b_t:
                    B.append(p[1])
            self.blocks.append(B)
        self.blocks = list(np.unique(np.array(self.blocks, dtype=object)))

    def run_frequency_blocking(self):
        self.__compute_blocking_tokens()
        self.__get_blocking_tokens()
        self.__compute_blocks()
