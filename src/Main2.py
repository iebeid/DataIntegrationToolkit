import pandas as pd
import scipy
import spacy


def main():
    articles_df =  pd.read_csv("A01_Articles_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    abstracts_df = pd.read_csv("A04_Abstract_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    print(articles_df.info())
    print(abstracts_df.info())
    extracted_abstracts = []
    model = "en_ner_bionlp13cg_md"
    for index,abstract in articles_df.iterrows():
        pmid = abstract[0]
        record = abstracts_df.loc[abstracts_df['PMID'] == pmid]
        abstract_extracted = ""
        for index2,abstract_part in record.iterrows():
            abstract_extracted = abstract_extracted + abstract_part[2]
            extracted_abstracts.append(abstract_extracted)

    for abstract in extracted_abstracts:
        nlp = spacy.load(model)
        doc = nlp(abstract)
        data_set = set([(X.text, X.label_) for X in doc.ents])


if __name__ == '__main__':
    main()

