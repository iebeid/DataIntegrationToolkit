import datetime
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

import modules.DWM10_Parms as DWM10_Parms
import modules.DWM14_BuildRefDict as DWM14_BuildRefDict
import modules.DWM15_BuildLinkIndex as DWM15_BuildLinkIndex
import modules.DWM16_BuildTokenFreqDict as DWM16_BuildTokenFreqDict
import modules.DWM42_BuildBlockPairs as DWM42_BuildBlockPairs
import modules.DWM55_LinkBlockPairs as DWM55_LinkBlockPairs
import modules.DWM96_WriteLinkIndex as DWM96_WriteLinkIndex

import modules.DWM100_GraphEntityProfiler as DWM100_GraphEntityProfiler
from modules.GraphEntityProfiler import DWM_GEP_EvaluatER
from modules.GraphEntityProfiler import DWM_GEP_StagER


def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper


@measure_execution_time
def main():
    version = 2.21
    now = datetime.datetime.now()
    tag = str(now.year) + (str(now.month)).zfill(2) + (str(now.day)).zfill(2)
    tag = tag + "_" + (str(now.hour)).zfill(2) + "_" + (str(now.minute)).zfill(2)
    log_file_name = "output/logs/DWM_Log_" + tag + ".txt"
    logFile = open(log_file_name, "w")
    parmFileName = "config/S1G-params.txt"
    # parmFileName = "config/S2G-params.txt"
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
    # parmFileName = "config/S20ACMDBLP-params.txt"
    # parmFileName = "config/S21DBLPGoogle-params.txt"
    # parmFileName = "config/S22Affiliations-params.txt"
    # parmFileName = "config/S23AmazonGoogle-params.txt"
    # parmFileName = "config/S24AbtRakuten-params.txt"
    # parmFileName = input("Enter Parameter File Name->")
    print("Graph Data Washing Machine (GDWM)", version)
    print("Graph Data Washing Machine (GDWM)", version, file=logFile)
    print("Date/Time", tag)
    print("Data/Time", tag, file=logFile)
    print("Running configuration file: " + parmFileName)
    print("Running configuration file: " + parmFileName, file=logFile)
    #----LOAD PARAMETERS----#
    DWM10_Parms.getParms(parmFileName, logFile)
    # ----TOKENIZATION----#
    refDict = DWM14_BuildRefDict.tokenizeInput()
    linkIndex = DWM15_BuildLinkIndex.buildLinkIndex(refDict)
    tokenFreqDict = DWM16_BuildTokenFreqDict.buildTokenFreqDict(refDict)
    #---BLOCKING----#
    blockPairList = DWM42_BuildBlockPairs.buildBlockPairs(refDict, linkIndex, tokenFreqDict)
    print("-------------- GRAPH ENTITY PROFILER INITIATED --------------")
    print("-------------- GRAPH ENTITY PROFILER INITIATED --------------", file=DWM10_Parms.logFile)
    er_stager = DWM_GEP_StagER.Stager(refDict, blockPairList, tokenFreqDict)
    # # Configuration 1
    # graph_object = er_stager.spek_block_linker()
    # graph_object = DWM100_GraphEntityProfiler.spek_model(graph_object)
    # Configuration 2
    # graph_object = er_stager.linkBlockPairs(blockPairList, refDict, tokenFreqDict)
    graph_object = er_stager.discrimininator_block_linker()
    graph_object.info()
    # graph_object = DWM100_GraphEntityProfiler.run_modularity_optimizaton_model(graph_object)
    graph_object = DWM100_GraphEntityProfiler.run_modularity_embedding_model(graph_object)
    # graph_object = DWM100_GraphEntityProfiler.run_spek_model_2(refDict, edge_list)
    # graph_object = DWM100_GraphEntityProfiler.run_spek_model(refDict, edge_list)
    # graph_object = DWM100_GraphEntityProfiler.run_spek_model_3(refDict, linkedPairList)
    # graph_object = DWM100_GraphEntityProfiler.run_spek_model_trial_4(refDict, edge_list)
    print("-------------- GRAPH ENTITY PROFILER COMPLETED --------------")
    print("-------------- GRAPH ENTITY PROFILER COMPLETED --------------", file=DWM10_Parms.logFile)
    # Step 4: final evaluation
    print("-------------- FINAL EVALUATION --------------")
    print("-------------- FINAL EVALUATION --------------", file=DWM10_Parms.logFile)
    graph_object.convert_node_labels_to_link_index()
    evaluator_final = DWM_GEP_EvaluatER.Evaluater(graph_object, DWM10_Parms.truthFileName)
    evaluator_final.run()
    # ------------------------------
    linkIndex = graph_object.link_index_dict
    # write Link Index to text file
    DWM96_WriteLinkIndex.writeLinkIndex(linkIndex, refDict)
    logFile.close()


if __name__ == '__main__':
    main()
