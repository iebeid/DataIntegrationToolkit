
import pandas as pd


def map_to_letter(number):
    map_value = 0
    if number == '0':
        map_value = 'a'
    if number == '1':
        map_value = 'b'
    if number == '2':
        map_value = 'c'
    if number == '3':
        map_value = 'd'
    if number == '4':
        map_value = 'e'
    if number == '5':
        map_value = 'f'
    if number == '6':
        map_value = 'g'
    if number == '7':
        map_value = 'h'
    if number == '8':
        map_value = 'i'
    if number == '9':
        map_value = 'j'
    return map_value

def map_code_to_letter(code):
    map_value = ''
    for c in code:
        single_map_value = map_to_letter(c)
        map_value = map_value + str(single_map_value)
    return "xxx"+map_value

def map_to_number(number):
    map_value = 0
    if number == 'a':
        map_value = '0'
    if number == 'b':
        map_value = '1'
    if number == 'c':
        map_value = '2'
    if number == 'd':
        map_value = '3'
    if number == 'e':
        map_value = '4'
    if number == 'f':
        map_value = '5'
    if number == 'g':
        map_value = '6'
    if number == 'h':
        map_value = '7'
    if number == 'i':
        map_value = '8'
    if number == 'j':
        map_value = '9'
    return map_value

def map_code_to_number(code):
    map_value = ''
    for c in code:
        single_map_value = map_to_number(c)
        map_value = map_value + str(single_map_value)
    return int(map_value)


def main():
    articles_df =  pd.read_csv("A01_Articles_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    abstracts_df = pd.read_csv("A04_Abstract_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    bioentities_df = pd.read_csv("B10_BERN_Main_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    print(articles_df.info())
    print(abstracts_df.info())
    print(bioentities_df.info())
    # Cleanup the abstracts data frame as one article might have several abstracts
    extracted_abstracts = []
    pmids = []
    just_one_abstract = ""
    just_one_pmid = ""
    for index,abstract in articles_df.iterrows():
        pmid = abstract[0]
        title = abstract[24]
        length_of_title = len(title)
        pmids.append(pmid)
        record = abstracts_df.loc[abstracts_df['PMID'] == pmid]

        abstract_extracted = ""
        for index2,abstract_part in record.iterrows():
            abstract_extracted = abstract_extracted + abstract_part[2]
        just_one_abstract = abstract_extracted
        just_one_pmid = pmid

        # Replace bioentities with code
        extracted_bioentities_df = bioentities_df[bioentities_df["PMID"]==pmid]
        if not extracted_bioentities_df.empty:
            for be in extracted_bioentities_df.iterrows():
                entity_id = str(be[1]["EntityID"])
                if entity_id != "0":
                    encoded_entity_id = map_code_to_letter(str(entity_id))
                    mention = str(be[1]["Mention"])
                    abstract_extracted = abstract_extracted.replace(mention, encoded_entity_id)
                    extracted_abstracts.append(abstract_extracted)

    # print("-----------------")
    # print(abstract_extracted)
    # print("-----------------")
    print("-----------------")
    print(just_one_pmid)
    print(just_one_abstract)
    print("-----------------")

    # tokenizer = AutoTokenizer.from_pretrained("scibert_scivocab_uncased")
    # model = AutoModel.from_pretrained("scibert_scivocab_uncased")
    # inputs = tokenizer("Hello world!", return_tensors="pt")
    # outputs = model(**inputs)
    # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    # inputs = tokenizer(just_one_abstract, return_tensors="pt")
    # outputs = model(**inputs)
    # print(outputs)







if __name__ == '__main__':
    main()