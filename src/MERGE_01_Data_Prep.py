

def data_prep(articles_df, abstracts_df):
    # print("Staring data preparation")
    corpus = {}
    for index,abstract in articles_df.iterrows():
        pmid = abstract[0]
        record = abstracts_df.loc[abstracts_df['PMID'] == pmid]
        abstract_extracted = ""
        for index2,abstract_part in record.iterrows():
            abstract_extracted = abstract_extracted + abstract_part[2]
        if len(abstract_extracted) != 0:
            corpus[int(pmid)] = str(abstract_extracted)
    return corpus