from tqdm import tqdm
import re


def remove_punctuation(token):
    # initializing punctuations string
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    # Removing punctuations in string
    for ele in token:
        if ele in punc:
            token = token.replace(ele, " ")
    return token


def process_token(token):
    processed_token = re.sub(' +', ' ', remove_punctuation(str(token).upper()))
    na = ["NAN","NULL","N/A"]
    if processed_token in na:
        processed_token = processed_token.replace(processed_token, " ")
    return str(processed_token)


def tokenize(data_frame):
    reference_data = {}
    for index, row in tqdm(data_frame.iterrows()):
        recID = process_token(row[0])
        fname = process_token(row[1])
        lname = process_token(row[2])
        mname = process_token(row[3])
        address = process_token(row[4])
        city = process_token(row[5])
        state = process_token(row[6])
        zipcode = process_token(row[7])
        ssn = process_token(row[8])
        reference_data[recID] = fname + " " + mname + " " + lname + " " + address + " " + city + " " + state + " " + zipcode + " " + ssn
    return reference_data

def process_single_reference(data_frame):
    reference_data = {}
    for index, row in tqdm(data_frame.iterrows()):
        recID = process_token(row[0])
        ref = process_token(row[1])
        reference_data[recID] = ref
    return reference_data