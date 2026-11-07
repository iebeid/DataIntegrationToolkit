import pandas as pd
import string
import nltk
# nltk.download('averaged_perceptron_tagger')
import spacy

sp = spacy.load('en_core_web_sm')

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def main():
    abstracts_df = pd.read_csv("A04_Abstract_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    print(abstracts_df.info())

    extracted_abstracts = []
    for index,abstract in abstracts_df.iterrows():
        pmid = abstract[1]
        record = abstracts_df.loc[abstracts_df['PMID'] == pmid]
        abstract_extracted = ""
        for index2,abstract_part in record.iterrows():
            abstract_extracted = abstract_extracted + abstract_part[2]

        #Cleanup string
        abstract_extracted = str(pmid) + " " + abstract_extracted.lower()
        # abstract_extracted = abstract_extracted.translate(str.maketrans('', '', string.punctuation))

        # print(pmid)
        # print(abstract_extracted)

        extracted_abstracts.append(abstract_extracted)


    print("Size before: " + str(len(extracted_abstracts)))

    # Remove duplicates from a list
    cleaned_extracted_abstracts = []
    [cleaned_extracted_abstracts.append(x) for x in extracted_abstracts if x not in cleaned_extracted_abstracts]

    # print(cleaned_extracted_abstracts)
    print("Size after: " + str(len(cleaned_extracted_abstracts)))

    # tokenize to sentence then to words
    tokenized_abstract_corpus = []
    for abs in cleaned_extracted_abstracts:
        abs_list = []
        sentences = abs.split(".")
        for sentence in sentences:
            words = sentence.split()
            tagged_words = nltk.pos_tag(words)
            if len(words) != 0:
                abs_list.append(tagged_words)
        tokenized_abstract_corpus.append(abs_list)

    # print(tokenized_abstract_corpus)

    for abs in cleaned_extracted_abstracts:
        tagged_abstract = sp(abs)
        for word in tagged_abstract:
            print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    main()
