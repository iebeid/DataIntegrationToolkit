import hashlib
import struct

import Levenshtein
from pylab import *
from textdistance import DamerauLevenshtein


class SimilER():

    def __init__(self, record_index, corpus, lsh_permutations, encoding):
        print("----SIMILARITY----")
        self.record_index = record_index
        self.corpus = corpus
        self.hash_table = {}
        self.lsh_permutations = lsh_permutations
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._max_hash = np.uint64((1 << 32) - 1)
        self.encoding = encoding
        self.hashvalues = np.ones(self.lsh_permutations, dtype=np.uint64) * self._max_hash
        gen = np.random.RandomState(1)
        self.permutations = np.array([(gen.randint(1, self._mersenne_prime, dtype=np.uint64),
                                       gen.randint(0, self._mersenne_prime, dtype=np.uint64)) for _ in
                                      range(self.lsh_permutations)], dtype=np.uint64).T

    def __sha1_hash32(self, data):
        return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

    def run_locality_sensitive_hashing(self):
        for id, document in enumerate(self.corpus):
            for d in document:
                hv = self.__sha1_hash32(d.encode(self.encoding))
                coeff_A, coeff_B = self.permutations
                phv = np.bitwise_and((coeff_A * hv + coeff_B) % self._mersenne_prime, self._max_hash)
                self.hashvalues = np.minimum(phv, self.hashvalues)
            self.hash_table[str(id)] = self.hashvalues

    def scoring_matrix(self, ref1, ref2):
        Class = DamerauLevenshtein()
        score = 0.0
        m = len(ref1)
        n = len(ref2)
        if m == 0 or n == 0:
            return score
        # generate m x n matrix
        matrix = [[0.0 for j in range(n)] for i in range(m)]
        maxVal = -1.0
        typo_threshold = 2
        discriminate_length = 5
        token_equivalence_condition = lambda word1, word2: (
                (word1 == word2) or (word1 in word2) or (word2 in word1) or (
                (Levenshtein.distance(word1, word2)) < typo_threshold))
        # populate matrix with similarities between tokens
        for j in range(0, m):
            token1 = ref1[j]
            for k in range(0, n):
                token2 = ref2[k]
                simVal = 0.0
                if token_equivalence_condition(token1, token2):
                    simVal = 0.5
                    matrix[j][k] = simVal
                    continue
                # Default Rule, otherwise use Damerau-levesthein distance
                simVal = Class.normalized_similarity(token1, token2)
                matrix[j][k] = simVal
        # end of matrix population
        loops = 0
        total = 0.0
        while True:
            maxVal = -1.0
            # search for maximum value in matrix
            for j in range(m):
                for k in range(n):
                    if matrix[j][k] > maxVal:
                        maxVal = matrix[j][k]
                        saveJ = j
                        saveK = k
            if maxVal < 0:
                return score
            total = total + maxVal
            loops += 1
            score = total / loops
            # set column saveK values to -1.0
            for j in range(m):
                matrix[j][saveK] = -1.0
            # set row saveJ values to -1.0
            for k in range(n):
                matrix[saveJ][k] = -1.0
                # end of while loop
        return score

    def scoring_matrix_std(self, ref1, ref2):
        Class = DamerauLevenshtein()
        mu = 0.0
        score = 0.0
        m = len(ref1)
        n = len(ref2)
        if m == 0 or n == 0:
            return score
        matrix = [[0.0 for j in range(n)] for i in range(m)]
        maxVal = -1.0
        for j in range(0, m):
            token1 = ref1[j]
            for k in range(0, n):
                token2 = ref2[k]
                simVal = 0.0
                if token1.isdigit() and token2.isdigit():
                    if token1 == token2:
                        simVal = 1.0
                    else:
                        simVal = 0.0
                    matrix[j][k] = simVal
                    continue
                if len(token1) == 1 or len(token2) == 1:
                    if token1 == token2:
                        simVal = 1.0
                    else:
                        simVal = 0.0
                    matrix[j][k] = simVal
                    continue
                simVal = Class.normalized_similarity(token1, token2)
                matrix[j][k] = simVal
        loops = 0
        total = 0.0
        while True:
            maxVal = -1.0
            for j in range(m):
                for k in range(n):
                    if matrix[j][k] > maxVal:
                        maxVal = matrix[j][k]
                        saveJ = j
                        saveK = k
            if maxVal < 0:
                return score
            total = total + maxVal
            loops += 1
            score = total / loops
            if score < mu:
                return score
            for j in range(m):
                matrix[j][saveK] = -1.0
            for k in range(n):
                matrix[saveJ][k] = -1.0
        return score

    def jaccard(self, s1, s2):
        a = set(s1)
        b = set(s2)
        if len(a) == 0 or len(b) == 0:
            return 0
        return float(len(a & b) / len(a | b))

    def estimated_jaccard(self, rec1, rec2):
        hvs1 = self.hash_table[rec1]
        hvs2 = self.hash_table[rec2]
        est_jaccard = float(np.count_nonzero(hvs1 == hvs2)) / float(self.lsh_permutations)
        return est_jaccard

    def cluster_entropy(self, cluster):
        tokenCount = 0
        refCnt = len(cluster)
        for j in range(0, refCnt):
            tokenCount = tokenCount + len(cluster[j])
        baseProb = 1 / float(refCnt)
        base = -tokenCount * baseProb * math.log(baseProb, 2)
        entropy = 0.0
        clusterSize = len(cluster)
        for j in range(0, len(cluster) - 1):
            jList = cluster[j]
            for token in jList:
                cnt = 1
                for k in range(j + 1, len(cluster)):
                    if token in cluster[k]:
                        cnt += 1
                        cluster[k].remove(token)
                tokenProb = cnt / clusterSize
                term = -tokenProb * math.log(tokenProb, 2)
                entropy += term
                quality = 1.0 - entropy / base
                cnt = 0
        for token in cluster[clusterSize - 1]:
            tokenProb = 1.0 / clusterSize
            term = -tokenProb * math.log(tokenProb, 2)
            entropy += term
            quality = 1.0 - entropy / base
        quality = 1.0 - entropy / base
        return quality
