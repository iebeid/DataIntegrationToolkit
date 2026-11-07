import re

import chardet
import numpy as np


class StagER():
    def __init__(self, delimiter, input_file_name, percentile_of_stop_words, percentile_of_blocking_words,
                 percentile_of_discriminate_words_lengths):
        print("----STAGING----")
        self.delimiter = delimiter
        self.input_file_name = input_file_name
        self.percentile_of_stop_words = percentile_of_stop_words
        self.percentile_of_blocking_words = percentile_of_blocking_words
        self.percentile_of_discriminate_words_lengths = percentile_of_discriminate_words_lengths
        self.count = 0
        self.beta = 0
        self.sigma = 0
        self.delta = 0
        self.encoding = None

        self.word_count_dict = {}
        self.word_to_index={}
        self.word_length_dict = {}
        self.document_index = {}
        self.document_inverse_index = {}
        # self.tokenized_corpus = {}
        self.unique_tokenized_corpus = {}
        self.filtered_unique_tokenized_corpus = {}
        self.discriminative_word_corpus = {}
        self.non_discriminative_word_corpus = {}
        self.blocking_corpus = {}
        self.signature_corpus = {}

        with open(self.input_file_name, "rb") as rawdata:
            result = chardet.detect(rawdata.read())
            self.encoding = result["encoding"]
            if self.encoding == "ascii":
                self.encoding = "utf-8"

    def __tokenizer(self, string):
        string = string.strip()
        string = string.lower()
        string = string.replace(self.delimiter, ' ')
        tokenList = re.split('[\s]+', string)
        newList = []
        for token in tokenList:
            newToken = re.sub('[\W]+', '', token)
            if len(newToken) > 0:
                newList.append(newToken)
        return newList

    def __remove_stop_words(self, token_list):
        newList = []
        for token in token_list:
            token = token.strip()
            if token in self.word_count_dict:
                freq = self.word_count_dict[token]
                if freq <= self.sigma:
                    newList.append(token)
        return newList

    def __unique_word_set(self, token_list):
        newList = []
        for token in token_list:
            token = token.strip()
            newList.append(token)
        new_list = list(np.unique(np.array(newList)))
        return new_list

    def __discriminate_word_list(self, token_list):
        d_list = [x for x in token_list if len(x) >= self.delta]
        return d_list

    def __filter_word_list(self, token_list, filter_list):
        f_list = [x for x in token_list if x not in filter_list]
        return f_list

    def __blocking(self, token_list):
        blocking_words = []
        for word in token_list:
            wc = self.word_count_dict[word]
            if 2 <= wc <= self.beta:
                blocking_words.append(word)
        return blocking_words

    def generate_word_corpus(self):
        with open(self.input_file_name, "r", encoding=self.encoding) as input_file:
            next(input_file)
            unique_id = 0
            for document in input_file:
                document = document.strip()
                first_delimiter = document.find(self.delimiter)
                doc_id = document[0:first_delimiter]
                body = document[first_delimiter + 1:]
                word_list = self.__tokenizer(body)
                unique_word_list = self.__unique_word_set(word_list)
                self.document_index[str(unique_id)] = str(doc_id)
                self.document_inverse_index[str(doc_id)] = str(unique_id)
                self.unique_tokenized_corpus[str(unique_id)] = unique_word_list
                unique_id = unique_id + 1
            for id, doc_id in self.document_index.items():
                unique_words = self.unique_tokenized_corpus[id]
                for w1 in unique_words:
                    initial_count = unique_words.count(w1)
                    if w1 in self.word_count_dict:
                        self.word_count_dict[w1] = self.word_count_dict[w1] + initial_count
                    else:
                        self.word_count_dict[w1] = initial_count
                for w2 in unique_words:
                    if w2 not in self.word_length_dict:
                        self.word_length_dict[w2] = len(w2)
            self.beta = np.percentile(np.array(list(self.word_count_dict.values())), self.percentile_of_blocking_words)
            self.sigma = np.percentile(np.array(list(self.word_count_dict.values())), self.percentile_of_stop_words)
            self.delta = np.percentile(np.array(list(self.word_length_dict.values())),
                                       self.percentile_of_discriminate_words_lengths)
            for id, doc_id in self.document_index.items():
                unique_words = self.unique_tokenized_corpus[id]
                filtered_unique_words = self.__remove_stop_words(unique_words)
                discriminate_words = self.__discriminate_word_list(filtered_unique_words)
                discriminate_words.sort(reverse=False)
                discriminate_words.sort(key=len, reverse=True)
                non_discriminate_words = self.__filter_word_list(filtered_unique_words, discriminate_words)
                blocking_words = self.__blocking(filtered_unique_words)
                self.filtered_unique_tokenized_corpus[id] = filtered_unique_words
                self.discriminative_word_corpus[id] = discriminate_words
                self.non_discriminative_word_corpus[id] = non_discriminate_words
                self.blocking_corpus[id] = blocking_words
                self.signature_corpus[id] = discriminate_words
