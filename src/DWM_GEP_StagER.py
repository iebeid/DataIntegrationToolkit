import sys

from textdistance import Cosine
from textdistance import MongeElkan

import modules.DWM10_Parms as DWM10_Parms
import modules.DWM65_ScoringMatrixStd as DWM65_ScoringMatrixStd
import modules.DWM67_JaccardComparator as DWM67_JaccardComparator
import \
    modules.DWM69_HierarchicalDocumentComparatorWithDigitDiscriminator as DWM69_HierarchicalDocumentComparatorWithDigitDiscriminator
from modules.GraphEntityProfiler.DWM_GEP_GraphOG import GraphOG


class Stager:

    # Constructor
    def __init__(self, references, block_pairs, token_frequencies):
        print("-------------- STAGING --------------")
        print("-------------- STAGING --------------", file=DWM10_Parms.logFile)
        self.references = dict(sorted(references.items()))
        self.token_frequencies = token_frequencies
        self.block_pairs = block_pairs
        self.block_pairs.sort()

    # Define nested function for removing stop words
    @staticmethod
    def removeStopWords(tokenList, tokenFreqDict):
        newList = []
        # print('-- tokenList', tokenList)
        for token in tokenList:
            tokenLen = len(token)
            includeToken = True
            freq = tokenFreqDict[token]
            if freq >= DWM10_Parms.sigma:
                includeToken = False
                # print('-- sigma rule', token, freq)
            if DWM10_Parms.removeExcludedBlkTokens:
                if tokenLen < DWM10_Parms.minBlkTokenLen:
                    includeToken = False
                    # print('-- min len rule', token, tokenLen)
                if token.isdigit() and DWM10_Parms.excludeNumericBlocks:
                    includeToken = False
                    # print('-- number rule', token)
            if DWM10_Parms.removeDuplicateTokens and (token in newList):
                includeToken = False
                # print('-- duplicate token rule', token)
            if includeToken:
                newList.append(token)
        # print('-- newList', newList)
        return newList

    def linkBlockPairs(self, blockPairList, refDict, tokenFreqDict):
        logFile = DWM10_Parms.logFile
        sigma = DWM10_Parms.sigma
        removeDuplicateTokens = DWM10_Parms.removeDuplicateTokens
        removeExcludedBlkTokens = DWM10_Parms.removeExcludedBlkTokens
        print('\n>>Starting DWM55')
        print('\n>>Starting DWM55', file=logFile)
        print('Sigma =', sigma)
        print('Sigma =', sigma, file=logFile)
        print('Remove Duplicate Tokens =', removeDuplicateTokens)
        print('Remove Duplicate Tokens =', removeDuplicateTokens, file=logFile)
        print('Remove Excluded Block Tokens =', removeExcludedBlkTokens)
        print('Remove Excluded Block Tokens =', removeExcludedBlkTokens, file=logFile)
        # end of nexted fucntion
        # Check for valid comparator
        validComparator = False
        comparator = DWM10_Parms.comparator
        if comparator == 'MongeElkan':
            Class = MongeElkan()
            validComparator = True
        if comparator == 'Cosine':
            Class = Cosine()
            validComparator = True
        if comparator == 'ScoringMatrixStd':
            Class = DWM65_ScoringMatrixStd
            validComparator = True
        if not validComparator:
            print('**Error: Invalid Comparator Value in Parms File', comparator)
            sys.exit()
        mu = DWM10_Parms.mu
        linkedPairList = []
        blockPairListLen = len(blockPairList)
        for j in range(0, blockPairListLen):
            pair = blockPairList[j]
            refIDs = pair.split('|')
            refID1 = refIDs[0]
            refID2 = refIDs[1]
            tokenList1 = self.removeStopWords(refDict[refID1], tokenFreqDict)
            tokenList2 = self.removeStopWords(refDict[refID2], tokenFreqDict)
            result = Class.normalized_similarity(tokenList1[:], tokenList2[:])
            if result >= mu:
                linkedPairList.append((refID1, refID2, result))
        print('Number of Pairs Linked =', len(linkedPairList), 'at mu=', mu)
        print('Number of Pairs Linked =', len(linkedPairList), 'at mu=', mu, file=logFile)
        # construct node label dict randomly assigned
        nodes = {}
        l = 0
        for n, v in refDict.items():
            nodes[str(n)] = {"type": "r", "attribute": list(v), "label": int(l)}
            l = l + 1
        return GraphOG(linkedPairList, nodes=nodes, undirected=True, weighted=True, link_single_nodes=True)

    def spek_jaccard_block_linker(self):
        edges = []
        for row in self.block_pairs:
            tokenized_row = row.split("|")
            s_n = tokenized_row[0]
            t_n = tokenized_row[1]
            s_ref = self.references[s_n]
            t_ref = self.references[t_n]
            p = DWM67_JaccardComparator.normalized_similarity(s_ref, t_ref)
            if p > 0:
                edges.append((s_n, t_n, p))
        nodes = {}
        l = 0, 0
        for n, att in self.references.items():
            nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
            l = l + 1
        return GraphOG(edges, nodes=nodes, undirected=True, weighted=True)

    def spek_scoring_matrix_block_linker(self):
        edges = []
        for row in self.block_pairs:
            tokenized_row = row.split("|")
            s_n = tokenized_row[0]
            t_n = tokenized_row[1]
            s_ref = self.references[s_n]
            t_ref = self.references[t_n]
            s_ref_filtered = []
            t_ref_filtered = []
            for t_s in s_ref:
                if t_s not in t_ref:
                    s_ref_filtered.append(t_s)
            for t_t in t_ref:
                if t_t not in s_ref:
                    t_ref_filtered.append(t_t)
            p = DWM65_ScoringMatrixStd.normalized_similarity(s_ref_filtered, t_ref_filtered)
            if p > 0:
                edges.append((s_n, t_n, p))
            else:
                edges.append((s_n, t_n, 1.0))
        nodes = {}
        l = 0
        for n, att in self.references.items():
            nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
            l = l + 1
        return nodes, edges

    def spek_block_linker(self):
        edges = []
        for row in self.block_pairs:
            tokenized_row = row.split("|")
            s_n = tokenized_row[0]
            t_n = tokenized_row[1]
            edges.append((s_n, t_n, 1.0))
        nodes = {}
        l = 0
        for n, att in self.references.items():
            nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
            l = l + 1
        return GraphOG(edges, nodes=nodes, undirected=True, weighted=True, link_single_nodes=False)

    def discrimininator_block_linker(self):
        edges = []
        for row in self.block_pairs:
            split_row = row.split("|")
            s_n = split_row[0]
            t_n = split_row[1]
            s_ref = self.references[s_n]
            t_ref = self.references[t_n]
            p = DWM69_HierarchicalDocumentComparatorWithDigitDiscriminator.decision(s_ref, t_ref, False, True,
                                                                                    self.token_frequencies,
                                                                                    character_error_tolerance=2,
                                                                                    digit_tolerance=6)
            if p != 0:
                edges.append((s_n, t_n, p))
        nodes = {}
        l = 0
        for n, att in self.references.items():
            nodes[str(n)] = {"type": "r", "attribute": list(att), "label": int(l)}
            l = l + 1
        return GraphOG(edges, nodes=nodes, undirected=True, weighted=True, link_single_nodes=True)
