import pandas as pd
import MERGE_01_Data_Prep
import MERGE_02_BioBERT_Extract
import MERGE_03_Spacy_Extract
import MERGE_04_Normalize_BioentityIDs
import MERGE_05_Map_Bioentities_To_Chem2Bio2RDF
import MERGE_06_Construct_And_Merge_PubMed_C2B2RDF_Graphs
import MERGE_07_Extract_Graph_Metadata
import MERGE_08_Graph_Embedding
import MERGE_09_Author_Profiler
import pickle

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def main():
    articles_df = pd.read_csv("data/A01_Articles_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    abstracts_df = pd.read_csv("data/A04_Abstract_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    bioentities_df = pd.read_csv("data/B10_BERN_Main_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    article_author_dataframe = pd.read_csv("data/A02_AuthorList_Cancer_50.csv", sep=",", encoding='utf-8', usecols=["PMID","AID"])
    citations_dataframe = pd.read_csv("data/B15_WOS_Citation_Top_50_Cancer_Inter.csv", sep=",", encoding='utf-8', usecols=["PMID","RefArticleId"])
    c2b2r_node_types = pd.read_csv("data/node2class.txt", sep="\t", encoding='utf-8', header=None)
    print(articles_df.info())
    print(abstracts_df.info())
    print(bioentities_df.info())
    # Data preparation
    corpus = MERGE_01_Data_Prep.data_prep(articles_df, abstracts_df)
    # Extract BioBERT entities
    bioentities_articles_dictionary, unique_bioentities_dictionary = MERGE_02_BioBERT_Extract.biobert_extract(corpus,
                                                                                                              bioentities_df)
    # # Extract Spacy entities
    # MERGE_03_Spacy_Extract.spacy_extract(corpus, bioentities_articles_dictionary)
    pretty(unique_bioentities_dictionary)
    MERGE_04_Normalize_BioentityIDs.normalize_bioentity_ids(unique_bioentities_dictionary)
    drug_mapping = pd.read_csv("drug-list.txt", sep="\t", encoding='utf-8', dtype=object)
    gene_mapping = pd.read_csv("gene-list.txt", sep="\t", encoding='utf-8', dtype=object)
    chem2bio2rdf_dataset = pd.read_csv("data/chem2bio2rdf.txt", sep="\t", encoding='utf-8', header=None)
    c2b2r_mapping_dict = MERGE_05_Map_Bioentities_To_Chem2Bio2RDF.map_to_c2b2r(unique_bioentities_dictionary,
                                                                               drug_mapping, gene_mapping,
                                                                               chem2bio2rdf_dataset)
    article_author_mapping = MERGE_05_Map_Bioentities_To_Chem2Bio2RDF.map_articles_to_authors(article_author_dataframe)
    citations_mapping = MERGE_05_Map_Bioentities_To_Chem2Bio2RDF.map_citations(citations_dataframe)
    # Construct graph
    graph = MERGE_06_Construct_And_Merge_PubMed_C2B2RDF_Graphs.construct_merge_graphs(article_author_mapping,
                                                                                      bioentities_articles_dictionary,
                                                                                      c2b2r_mapping_dict,
                                                                                      citations_mapping,
                                                                                      chem2bio2rdf_dataset)
    print(graph)
    print(len(graph))
    # open a file, where you ant to store the data
    file = open('graph-triples.npy', 'wb')
    # dump information to that file
    pickle.dump(graph, file)
    # close the file
    file.close()
    stellargraph_model = 'stellargraph-model.npy'
    c2b2r_node_types = pd.read_csv("data/node2class.txt", sep="\t", encoding='utf-8', header=None)
    graph_metadata = MERGE_07_Extract_Graph_Metadata.extract_graph_metadata('graph-triples.npy', c2b2r_node_types)
    sg_graph = MERGE_08_Graph_Embedding.construct_sg_graph('graph-triples.npy', graph_metadata, stellargraph_model)
    print(sg_graph.info())
    MERGE_08_Graph_Embedding.embed_graph_node2vec(stellargraph_model)
    MERGE_08_Graph_Embedding.embed_graph_metapath2vec(stellargraph_model)
    # At this point the output
    # model_file = "metapath2vec.npy"
    stellargraph_model = "stellargraph-model.npy"
    # model_file = "pubmed2c2b2r.node2vec.npy"
    model_file = "metapath2vec.npy"
    # MERGE_08_Graph_Embedding.analyze_model(stellargraph_model, model_file)
    MERGE_08_Graph_Embedding.visualize(stellargraph_model, model_file)
    MERGE_09_Author_Profiler.similarity_analysis(stellargraph_model, model_file)
    MERGE_09_Author_Profiler.most_influential(stellargraph_model, model_file)


if __name__ == '__main__':
    main()