#!/usr/bin/env python
# coding: utf-8
# In[1]:

import datetime
import time

import modules.DWM10_Parms as DWM10_Parms
import modules.DWM14_BuildRefDict as DWM14_BuildRefDict
import modules.DWM15_BuildLinkIndex as DWM15_BuildLinkIndex
import modules.DWM16_BuildTokenFreqDict as DWM16_BuildTokenFreqDict
import modules.DWM25_GlobalTokenReplace as DWM25_GlobalTokenReplace
import modules.DWM42_BuildBlockPairs as DWM42_BuildBlockPairs
import modules.DWM55_LinkBlockPairs as DWM55_LinkBlockPairs
import modules.DWM80_TransitiveClosure as DWM80_TransitiveClosure
import modules.DWM90_IterateClusters as DWM90_IterateClusters
import modules.DWM96_WriteLinkIndex as DWM96_WriteLinkIndex
import modules.DWM97_ClusterProfile as DWM97_ClusterProfile
import modules.DWM99_ERmetrics as DWM99_ERmetrics


def main():
    version = 2.21
    # get start time for timer
    startTime = time.time()
    # date time is used to label the logfile
    now = datetime.datetime.now()
    tag = str(now.year) + (str(now.month)).zfill(2) + (str(now.day)).zfill(2)
    tag = tag + '_' + (str(now.hour)).zfill(2) + '_' + (str(now.minute)).zfill(2)
    logFile = open('DWM_Log_' + tag + '.txt', 'w')
    print("Data Washing Machine Refactor Version", version)
    print("Data Washing Machine Refactor Version", version, file=logFile)
    print("Date/Time", tag)
    print("Data/Time", tag, file=logFile)
    now1 = datetime.datetime.now()
    # parmFileName = "config/S1G-params.txt"
    parmFileName = "config/S2G-params.txt"
    # parmFileName = "config/S3Rest-params.txt"
    # parmFileName = "config/S4G-params.txt"
    # parmFileName = "config/S5G-params.txt"
    # parmFileName = "config/S6GeCo-params.txt"
    # parmFileName = "config/S7GX-params.txt"
    # parmFileName = "config/S8P-params.txt"
    # parmFileName = "config/S9P-params.txt"
    # parmFileName = "config/S10PX-params.txt"
    # parmFileName = "config/S11PX-params.txt"
    # parmFileName = "config/S12PX-params.txt"
    # parmFileName = "config/S13GX-params.txt"
    # parmFileName = "config/S14GX-params.txt"
    # parmFileName = "config/S15GX-params.txt"
    # parmFileName = "config/S16PX-params.txt"
    # parmFileName = "config/S17PX-params.txt"
    # parmFileName = "config/S18PX-params.txt"
    # parmFileName = input('Enter Parameter File Name->')
    print('\n\nRunning parms file', parmFileName)
    print('\nRunning parms file ', parmFileName, file=logFile)
    DWM10_Parms.getParms(parmFileName, logFile)
    # Must get mu start and save to muStart for single parms file
    # and epsilonStart for parmeter single parms file
    # DWM10_Parms.blockCorrection changes if DWM45 runs need original value for reporting
    DWM10_Parms.muStart = DWM10_Parms.mu
    DWM10_Parms.epsilonStart = DWM10_Parms.epsilon
    DWM10_Parms.blockCorrect = DWM10_Parms.blockCorrection
    # Create refDict, a dictionary where key=refID, value is list of reference tokens
    refDict = DWM14_BuildRefDict.tokenizeInput()
    # Create linkIndx, a dictionary where key=refID, value is cluster ID`
    linkIndex = DWM15_BuildLinkIndex.buildLinkIndex(refDict)
    # Create tokenFeqDict, a dictionary where key=token, value is token frequency
    tokenFreqDict = DWM16_BuildTokenFreqDict.buildTokenFreqDict(refDict)
    # create dictionary of corrections (stdTokenDict), leave empty if not running replacement
    # if global replacement configured, populate stdTokenDict of corrections in DWM25
    if DWM10_Parms.runGlobalCorrection:
        refDict = DWM25_GlobalTokenReplace.globalReplace(refDict, tokenFreqDict)
        tokenFreqDict = DWM16_BuildTokenFreqDict.buildTokenFreqDict(refDict)
    print('\n>>Starting Iterations')
    print('\n>>Starting Iterations', file=logFile)
    mu = DWM10_Parms.mu
    print('mu start value=', mu)
    print('mu start value=', mu, file=logFile)
    muIterate = DWM10_Parms.muIterate
    print('mu iterate value=', muIterate)
    print('mu iterate value=', muIterate, file=logFile)
    epsilon = DWM10_Parms.epsilon
    print('epsilon start value=', epsilon)
    print('epsilon start value=', epsilon, file=logFile)
    epsilonIterate = DWM10_Parms.epsilonIterate
    print('epsilon iterate value=', epsilonIterate)
    print('epsilon iterate value=', epsilonIterate, file=logFile)
    comparator = DWM10_Parms.comparator
    print('comparator =', comparator)
    print('comparator =', comparator, file=logFile)
    firstIteration = True
    print('\n****New Iteration\nSize of refDict =', len(refDict))
    print('\n****New Iteration\nSize of refDict =', len(refDict), file=logFile)
    # blockList = DWM40_BuildBlocks.buildBlocks(logFile, refList, tokenFreqDict)
    blockPairList = DWM42_BuildBlockPairs.buildBlockPairs(refDict, linkIndex, tokenFreqDict)
    if len(blockPairList) == 0:
        print('--Ending because blockPairList is empty')
        print('--Ending because blockPairList is empty', file=logFile)
        return
    linkedPairList = DWM55_LinkBlockPairs.linkBlockPairs(blockPairList, refDict, tokenFreqDict)
    if len(linkedPairList) == 0:
        print('Ending because linkedPairList is empty')
        print('Ending because linkedPairList is empty', file=logFile)
        return
    # add single nodes to the pair list
    # import numpy as np
    # all_nodes = []
    # for e in linkedPairList:
    #     all_nodes.append(e[0])
    #     all_nodes.append(e[1])
    # unique_records = list(np.unique(np.array(all_nodes)))
    # difference = refDict.keys() - unique_records
    # for r in difference:
    #     linkedPairList.append((r,r,1))
    # # save the pairs to file
    # linkedPairsFile = open("S2-weightedLinkedPairList-01-including-single.npy", "wb")
    # pickle.dump(linkedPairList, linkedPairsFile)
    # linkedPairsFile.close()
    # refDictFile = open("S2-refDict.npy", "wb")
    # pickle.dump(refDict, refDictFile)
    # refDictFile.close()
    clusterList = DWM80_TransitiveClosure.transitiveClosure(linkedPairList)
    if len(clusterList) == 0:
        print('--Ending because clusterList is empty')
        print('--Ending because clusterList is empty', file=logFile)
        return
    iterationLinkIndex = DWM90_IterateClusters.iterateClusters(clusterList, refDict, linkIndex)
    print("\n>>Itermediate Results from this Iteration")
    print("\n>>Itermediate Results from this Iteration", file=logFile)
    # Run iteration profile and statistics if requested
    if DWM10_Parms.runIterationProfile:
        DWM97_ClusterProfile.generateProfile(iterationLinkIndex)
        if DWM10_Parms.truthFileName != '':
            DWM99_ERmetrics.generateMetrics(iterationLinkIndex)
    print('\n>>End of Iteration, Resetting mu and epsilon')
    print('\n>>End of Iteration, Resetting mu and epsilon', file=logFile)
    mu += muIterate
    mu = round(mu, 2)
    DWM10_Parms.mu = mu
    print('>>>New Value of mu = ', mu)
    print('>>>New Value of mu = ', mu, file=logFile)
    epsilon += epsilonIterate
    epsilon = round(epsilon, 2)
    DWM10_Parms.epsilon = epsilon
    print('>>>New Value of epsilon = ', epsilon)
    print('>>>New Value of epsilon = ', epsilon, file=logFile)
    # End of iterations
    # write Link Index to text file
    DWM96_WriteLinkIndex.writeLinkIndex(linkIndex, refDict)
    # Generate Cluster Profile
    DWM97_ClusterProfile.generateProfile(linkIndex)
    # Generat ER Metrics if truthFileName was given
    if DWM10_Parms.truthFileName != '':
        # cluster_modularities = {}
        # DWM99_ERmetrics.generateMetricsFull(logFile, len(refDict.keys()), cluster_modularities, linkIndex,
        #                                 DWM10_Parms.truthFileName)
        print(linkIndex)
        DWM99_ERmetrics.generateMetricsBasic(len(refDict.keys()), linkIndex)
    now2 = datetime.datetime.now()
    print("\nTotal File Runtime =", now2 - now1, file=logFile)
    print("\nEnd of File ", parmFileName)
    print('Time to run File ', now2 - now1)
    print("End of File ", parmFileName, file=logFile)

    endTime = time.time()
    totalTime = endTime - startTime
    print("All Files Total Runtime =", totalTime / 60, " minutes")
    print("All Files Total Runtime =", totalTime / 60, " minutes", file=logFile)
    print("End of Program")
    print("End of Program", file=logFile)
    logFile.close()


if __name__ == '__main__':
    main()
