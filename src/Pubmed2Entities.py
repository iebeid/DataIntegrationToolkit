import pandas as pd
import scipy
import spacy
import pprint
from spacy import displacy





def display_entities(model, document):
    nlp = spacy.load(model)
    doc = nlp(document)
    # display_doc = displacy.render(doc, jupyter=True, style="ent")
    data_set = set([(X.text, X.label_) for X in doc.ents])
    print(len(data_set))
    print(data_set)
    # pp = pprint.PrettyPrinter(width=41, compact=True)
    # entity_and_label = pp.pprint(data_set)
    # displacy_image = displacy.render(next(doc.sents), jupyter=False, style="ent")

    # return display_doc, entity_and_label


def main():
    print("Start Program")
    articles_df = pd.read_csv("A01_Articles_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    abstracts_df = pd.read_csv("A04_Abstract_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    bioentities_df = pd.read_csv("B10_BERN_Main_Top_50_Cancer.csv", sep=",", encoding='utf-8')
    print(articles_df.info())
    print(abstracts_df.info())
    print(bioentities_df.info())
    test_doc = "Hepatocellular carcinoma appears frequently in patients with cirrhosis. " \
               "Surveillance by biannual ultrasound is recommended for such patients because it " \
               "allows diagnosis at an early stage, when effective therapies are feasible. The best " \
               "candidates for resection are patients with a solitary tumour and preserved liver function. " \
               "Liver transplantation benefits patients who are not good candidates for surgical resection, and " \
               "the best candidates are those within Milan criteria (solitary tumour ?5 cm or up to three nodules ?3 cm). " \
               "Image-guided ablation is the most frequently used therapeutic strategy, but its efficacy is limited by the size " \
               "of the tumour and its localisation. Chemoembolisation has survival benefit in asymptomatic patients with multifocal " \
               "disease without vascular invasion or extrahepatic spread. Finally, sorafenib, lenvatinib, which is non-inferior to sorafenib, " \
               "and regorafenib increase survival and are the standard treatments in advanced hepatocellular carcinoma. This Seminar summarises the " \
               "scientific evidence that supports the current recommendations for clinical practice, and discusses the areas in which more research is needed."

    display_entities("en_ner_bionlp13cg_md", test_doc)
    display_entities("en_ner_bc5cdr_md", test_doc)








if __name__ == '__main__':
    main()