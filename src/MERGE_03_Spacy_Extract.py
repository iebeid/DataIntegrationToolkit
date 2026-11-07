import spacy
import scispacy


def spacy_extract(corpus, bioentities_articles_dictionary):
    # print("Start Spacy Bioentity Extraction")

    for pmid in corpus:
        abstract = corpus[pmid]
        bioentity_information = bioentities_articles_dictionary[str(pmid)]
        nlp = spacy.load("en_ner_bionlp13cg_md")
        doc = nlp(abstract)
        for X in doc.ents:
            extracted_mention = X.text
            extracted_type = X.label_
            if (extracted_type in ["CANCER", "GENE_OR_GENE_PRODUCT", "SIMPLE_CHEMICAL"]):
                for entity_id in bioentity_information:
                    type = entity_id.split("/")[0]
                    actual_entity_id = entity_id.split("/")[1]
                    list_of_mentions = bioentity_information[entity_id]
                    # if extracted_mention not in list_of_mentions:
                    #     # There are mulitple possbilities in that case.
                    #     # First, it might be that it is a new entity.
                    #     # Second it might be that it is a new mention to an existing entity.







