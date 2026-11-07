import pandas as pd
import numpy as np
from collections import defaultdict


def biobert_extract(corpus, bioentities_df):
    # print("Start BioBERT Bioentity Extraction")
    bioentities_articles_dictionary = {}
    unique_bioentities_dictionary = {}
    for pmid in corpus:
        extracted_bioentities_df = bioentities_df[bioentities_df["PMID"] == pmid]
        bioentities_in_articles_dict = {}
        sorted_bioentities = {}
        for be in extracted_bioentities_df.iterrows():
            bioentity_id = str(be[1]["EntityID"])
            bioentity_type = str(be[1]["Type"])
            bioentity_mention = str(be[1]["Mention"])
            if (bioentity_id != "0") and (bioentity_type in ["2", "3", "4"]):
                typed_bioentity_id = ""
                if bioentity_type == "2":
                    typed_bioentity_id = str("disease/" + str(bioentity_id))
                if bioentity_type == "3":
                    typed_bioentity_id = str("gene/" + str(bioentity_id))
                if bioentity_type == "4":
                    typed_bioentity_id = str("drug/" + str(bioentity_id))
                bioentities_in_articles_dict[bioentity_mention] = typed_bioentity_id
                # Fillup bioentity id
                unique_bioentities_dictionary[bioentity_mention] = typed_bioentity_id
            sorted_bioentities = defaultdict(list)
            for key, val in sorted(bioentities_in_articles_dict.items()):
                sorted_bioentities[val].append(key)
            sorted_bioentities = dict(sorted_bioentities)

        bioentities_articles_dictionary[str(pmid)] = sorted_bioentities

    sorted_unique_bioentities = defaultdict(list)
    for key, val in sorted(unique_bioentities_dictionary.items()):
        sorted_unique_bioentities[val].append(key)
    sorted_unique_bioentities = dict(sorted_unique_bioentities)

    return bioentities_articles_dictionary, sorted_unique_bioentities
