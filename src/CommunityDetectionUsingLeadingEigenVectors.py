
from igraph import *


def main():
    karate_graph = Graph.Read_Ncol('karate_edges_77.txt',directed=False)
    print(karate_graph)
    print(karate_graph.is_weighted())
    edge_weights=[]
    for edge in karate_graph.es:
        edge_weights.append(edge["weight"])
    communities_newman = karate_graph.community_leading_eigenvector(weights=edge_weights)
    print(communities_newman)

if __name__ == '__main__':
    main()