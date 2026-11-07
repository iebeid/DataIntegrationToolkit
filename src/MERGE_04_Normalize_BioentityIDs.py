
def normalize_bioentity_ids(unique_bioentity_dictionary):
    # drugs
    final_drug_mention_list = []
    for given_id in unique_bioentity_dictionary:
        mention_list = unique_bioentity_dictionary[given_id]
        entity_type = given_id.split("/")[0]
        if entity_type == "gene":
            for m in mention_list:
                final_drug_mention_list.append(m)
    # print('\n'.join(final_drug_mention_list))

