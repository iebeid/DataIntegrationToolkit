import numpy as np
from collections import defaultdict

def map_to_c2b2r(unique_bioentities_dictionary, drug_mapping, gene_mapping, chem2bio2rdf_dataset):
    # print(drug_mapping)
    mapping_drug_biobert_entity_id_to_c2b2r = []
    mapping_gene_biobert_entity_id_to_c2b2r = []
    for index, r1 in drug_mapping.iterrows():
        drug_mention = str(r1[0])
        pubchem_id = str(r1[1])
        for be1 in unique_bioentities_dictionary:
            list_of_mentions = unique_bioentities_dictionary[be1]
            if drug_mention in list_of_mentions:
                # then map entity to c2b2r
                c2b2r_string_1 = "http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/" + pubchem_id
                mapping_drug_biobert_entity_id_to_c2b2r.append((be1, c2b2r_string_1))
    for index, r2 in gene_mapping.iterrows():
        gene_mention = str(r2[0])
        for be2 in unique_bioentities_dictionary:
            list_of_mentions_2 = unique_bioentities_dictionary[be2]
            if gene_mention in list_of_mentions_2:
                # then map entity to c2b2r
                c2b2r_string_2 = "http://chem2bio2rdf.org/uniprot/resource/gene/" + gene_mention.upper()
                mapping_gene_biobert_entity_id_to_c2b2r.append((be2, c2b2r_string_2))

    # print(mapping_drug_biobert_entity_id_to_c2b2r)
    # print(mapping_gene_biobert_entity_id_to_c2b2r)

    # Search for the entities in chem2bio2rdf if found then we keep the mapping record, else we remove it from the mapping
    mapping_biobert_entity_id_to_c2b2r = mapping_drug_biobert_entity_id_to_c2b2r + mapping_gene_biobert_entity_id_to_c2b2r
    source_nodes = []
    target_nodes = []
    for index, triple in chem2bio2rdf_dataset.iterrows():
        source_node = str(triple[0])
        edge_type = str(triple[1])
        target_node = str(triple[2])
        source_nodes.append(source_node)
        target_nodes.append(target_node)
    unique_nodes = np.unique(np.array(source_nodes + target_nodes))
    # print(str(len(mapping_biobert_entity_id_to_c2b2r)))
    # print(unique_nodes)
    final_list = {}
    for record in mapping_biobert_entity_id_to_c2b2r:
        # print(record[1])
        if record[1] in unique_nodes:
            final_list[record[0]] = record[1]
    # print(str(len(final_list)))
    # print(final_list)
    return final_list


def map_articles_to_authors(article_author_dataframe):
    article_author_mapping = {}
    for index, row in article_author_dataframe.iterrows():
        article = row[0]
        author = row[1]
        # print(article)
        # print(author)
        article_author_mapping[author] = article

    sorted_article_author_mapping = defaultdict(list)
    for key, val in sorted(article_author_mapping.items()):
        sorted_article_author_mapping[val].append(key)
    sorted_unique_bioentities = dict(sorted_article_author_mapping)
    return sorted_unique_bioentities


def map_citations(citations_dataframe):
    citations = {}
    for index, row in citations_dataframe.iterrows():
        print(row)
        source_article = str(row[0])
        target_article = str(row[1])
        citations[source_article] = target_article
    return citations