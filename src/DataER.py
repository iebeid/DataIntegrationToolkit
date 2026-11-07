import time
import re
import chardet
import numpy as np
import pandas as pd
import networkx as nx
from GraphOG import GraphOG

def measure_execution_time(func):
    def wrapper():
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        print("Total runtime: " + str(round((end_time - start_time), 4)) + " seconds")

    return wrapper


@measure_execution_time
def acm_dblp_main():
    with open('data/DBLP-ACM/ACM.csv', "rb") as rawdata:
        result = chardet.detect(rawdata.read())
        encoding1 = result["encoding"]
        if encoding1 == "ascii":
            encoding1 = "utf-8"
    with open('data/DBLP-ACM/DBLP2.csv', "rb") as rawdata:
        result = chardet.detect(rawdata.read())
        encoding2 = result["encoding"]
        if encoding2 == "ascii":
            encoding2 = "utf-8"
    df1 = pd.read_csv('data/DBLP-ACM/ACM.csv',encoding=encoding1)
    df2 = pd.read_csv('data/DBLP-ACM/DBLP2.csv',encoding=encoding2)
    frames = [df1, df2]
    result = pd.concat(frames)
    result.reset_index()
    print(result.head())
    result.to_csv('data/DBLP-ACM/S25DBLPACM-2.txt',index=False)

@measure_execution_time
def acm_dblp_truth():
    df = pd.read_csv('data/DBLP-ACM/DBLP-ACM_perfectMapping.csv')
    print(df.to_string())
    edges = []
    for index, row in df.iterrows():
        edges.append((str(row["idDBLP"]),str(row["idACM"]),1.0))
    graph = GraphOG(edges, undirected=True, weighted=True, link_single_nodes=True, simple_graph=True)
    graph.info(None)
    g = graph.to_networkx
    components = nx.connected_components(g)
    cluster_list = []
    for enum, cc in enumerate(components):
        for n in cc:
            print(str(n) + "," + str(enum))
            cluster_list.append([str(n),str(enum)])
    final_df = pd.DataFrame(cluster_list,
                      columns=['RecID', 'idtruth'])
    print(final_df.to_string())
    final_df.to_csv('data/DBLP-ACM/truthDBLPACM-2.txt',index=False)

def main():
    print("yeah")


if __name__ == '__main__':
    # acm_dblp_truth()
    # acm_dblp_main()
    main()