import pickle
import numpy as np
import pandas
from tqdm import tqdm


def extract_graph_metadata(graph_object_file, c2b2r_node_types):
    print(c2b2r_node_types)
    file = open(graph_object_file, 'rb')
    graph_triples = pickle.load(file)
    file.close()

    source_nodes = []
    target_nodes = []
    orientation = []
    for triple in graph_triples:
        source_nodes.append(triple[0])
        target_nodes.append(triple[2])
        orientation.append(triple[1])

    all_nodes = source_nodes + target_nodes
    nodes = np.unique(np.array(all_nodes)).tolist()

    graph_metadata = {}
    print(str(len(nodes)))
    for n in tqdm(nodes):
        parts = n.split("/")
        node_type = "None"
        if parts[0] == "http:":
            found_df = c2b2r_node_types.loc[c2b2r_node_types[0] == n]


            found_node_type = found_df.iloc[0][1]

            # if found_node_type == "Gene ontology":
            #     node_type = "gene_ontology"
            # if found_node_type == "Chemical ontology":
            #     node_type = "chem_ontology"
            # if found_node_type == "Substructure":
            #     node_type = "sub_structure"
            if found_node_type == "Target":
                node_type = "gene"
            # if found_node_type == "Tissue":
            #     node_type = "tissue"
            # if found_node_type == "Pathway":
            #     node_type = "pathway"
            if found_node_type == "Disease":
                node_type = "disease"
            if found_node_type == "Chemical Compound/Drug":
                node_type = "drug"
            if found_node_type == "Side effect":
                node_type = "side_effect"
        if parts[0] == "article":
            node_type = "article"
        if parts[0] == "author":
            node_type = "author"
        if parts[0] == "bioentity":
            if parts[1] == "drug":
                node_type = "drug"
            if parts[1] == "gene":
                node_type = "gene"

        graph_metadata[n] = node_type

    print(str(len(graph_metadata)))


    return graph_metadata



