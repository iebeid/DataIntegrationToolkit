# Author: Islam Akef Ebeid
# Title: PubMed UpdatER

import os
import json
import xmltodict

class JournalIssue:
    pass

class Journal:
    pass



def main():
    print("Welcome to PubMed UpdatER")
    print("-------------------------")
    print("Description: This program is intended "
          "to load the latest PubMed data files into a relational database.")
    print("-------------------------")
    data_directory="../Data"
    # Get the current working directory. AI generated.
    current_directory = os.getcwd()
    # List the files in the current directory. AI generated.
    files = os.listdir(current_directory+"/"+data_directory)
    # Print the files. AI generated.
    for file in files:
        with open(data_directory+"/"+file,encoding="utf8") as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            result_set = data_dict['PubmedArticleSet']['PubmedArticle']
            for tag in result_set:
                journal_issn = tag["MedlineCitation"]["Article"]["Journal"]["ISSN"]["#text"]
                journal_title = tag["MedlineCitation"]["Article"]["Journal"]["Title"]
                journal_issue_volume = tag["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["Volume"]
                journal_issue_number = tag["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["Issue"]
                journal_issue_pubdate_year = tag["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]["Year"]
                journal_issue_pubdate_month = tag["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]["Month"]
                journal_issue_pubdate_day = tag["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]["Day"]


                print("Journal ISSN:", journal_title["ISSN"]["#text"])
                print("Journal Title:", journal_title["Title"])
                print("Journal Issue:", journal_title["JournalIssue"])
                print("PMID:", tag["MedlineCitation"]["PMID"]["#text"])
                print("Title:", tag["MedlineCitation"]["Article"]["ArticleTitle"].replace("[","").replace("]",""))
                try:
                    authors = tag["MedlineCitation"]["Article"]["AuthorList"]["Author"]
                    for author in authors:
                        print("Author:", author["ForeName"], author["LastName"])
                except:
                    print("No authors")
                try:
                    print("Abstract:", tag["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])
                except:
                    print("No abstract")
                print("---------------------------")

    print("End of program")



if __name__=="__main__":
    main()