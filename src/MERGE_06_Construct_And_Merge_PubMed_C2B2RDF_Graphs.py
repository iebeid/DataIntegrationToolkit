
from tqdm import tqdm

def construct_merge_graphs(article_author_mapping, article_bioentity_mapping, bioentity_c2b2r_mapping, citation_mapping, chem2bio2rdf):
    # print("Constructing PubMed Graph")
    graph = []

    linked_bioentities = []
    linked_articles = []
    chem2bio2rdf_entities = []
    c2b2r_intermediate_mapping = {}
    for be in tqdm(bioentity_c2b2r_mapping):
        c2b2r_entity = bioentity_c2b2r_mapping[be]
        source_node = "bioentity/" + str(be)
        relation = "sameAs"
        target_node = str(c2b2r_entity)
        linked_bioentities.append(source_node)
        chem2bio2rdf_entities.append(target_node)
        c2b2r_intermediate_mapping[target_node] = source_node
        # graph.append((source_node, relation, target_node))

    for pmid in tqdm(article_bioentity_mapping):
        bioentities_dict = article_bioentity_mapping[pmid]
        for be in bioentities_dict.keys():
            source_node = "bioentity/"+str(be)
            relation = "mentionedIn/mentions"
            target_node = "article/pmid/"+str(pmid)
            # if bioentity is linked to chem2bio2rdf
            if source_node in linked_bioentities:
                linked_articles.append(target_node)
                graph.append((source_node,relation,target_node))

    for pmid in tqdm(article_author_mapping):
        authors_dict = article_author_mapping[pmid]
        for a in authors_dict:
            source_node = "author/"+str(a)
            relation = "wrote/isWrittenBy"
            target_node = "article/pmid/"+str(pmid)
            if target_node in linked_articles:
                graph.append((source_node,relation,target_node))

    for pmid in tqdm(citation_mapping):
        linked_article = citation_mapping[pmid]
        # print(linked_article)
        source_node = "article/pmid/" + str(pmid)
        relation = "cites/citedBy"
        target_node = "article/pmid/" + str(linked_article)
        if (source_node in linked_articles) or (target_node in linked_articles):
            graph.append((source_node, relation, target_node))

    # Link chem2bio2rdf
    unwanted_relations = ["http://chem2bio2rdf.org/chebi", "http://chem2bio2rdf.org/chemogenomics/resource/expression",
                          "http://chem2bio2rdf.org/chemogenomics/resource/expression", "http://chem2bio2rdf.org/chebi",
                          "http://chem2bio2rdf.org/uniprot/resource/GO_ID", "http://chem2bio2rdf.org/substructure",
                          "http://chem2bio2rdf.org/hgnc/resource/Gene_Family_Name", "http://chem2bio2rdf.org/substructure",
                          "http://chem2bio2rdf.org/hprd/resource/tissue", "http://chem2bio2rdf.org/kegg/resource/protein"]

    c2b2r_source_nodes = []
    c2b2r_target_nodes = []
    for index, row in chem2bio2rdf.iterrows():
        c2b2r_source_nodes.append(row[0])
        c2b2r_target_nodes.append(row[2])

    for row2 in tqdm(chem2bio2rdf_entities):
        res_list = []
        if (row2 in c2b2r_source_nodes):
            res_list = [i for i in range(len(c2b2r_source_nodes)) if c2b2r_source_nodes[i] == row2]
        if (row2 in c2b2r_target_nodes):
            res_list = [i for i in range(len(c2b2r_target_nodes)) if c2b2r_target_nodes[i] == row2]
        selected_c2b2r_dataframe = chem2bio2rdf.iloc[res_list, :]
        for i,r in selected_c2b2r_dataframe.iterrows():
            if (r[0] in chem2bio2rdf_entities) and (r[2] in chem2bio2rdf_entities) and (r[1] not in unwanted_relations):
                graph.append((str(c2b2r_intermediate_mapping[r[0]]),str(r[1]),str(c2b2r_intermediate_mapping[r[2]])))
            elif (r[0] in chem2bio2rdf_entities) and (r[1] not in unwanted_relations):
                graph.append((str(c2b2r_intermediate_mapping[r[0]]),str(r[1]),str(r[2])))
            elif (r[2] in chem2bio2rdf_entities) and (r[1] not in unwanted_relations):
                graph.append((str(r[0]),str(r[1]),str(c2b2r_intermediate_mapping[r[2]])))
            elif r[1] not in unwanted_relations:
                graph.append((str(r[0]), str(r[1]), str(r[2])))










    return graph