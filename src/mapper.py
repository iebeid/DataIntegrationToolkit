from importer import *


class MappER():
    def __init__(self, delimiter, input_file_name):
        self.delimiter = delimiter
        self.input_file_name = input_file_name
        self.encoding = UtilityOperation.determine_file_encoding(self.input_file_name)

        # Relationships
        self.entities_to_tokens = {}
        self.tokens_to_entities = {}
        self.tokens_to_characters = {}
        self.characters_to_tokens = {}

        # Lookups
        self.entity_vocab = {}
        self.inverse_entity_vocab = {}  # Tells you how many initial entity profiles you have
        self.token_vocab = {}
        self.inverse_token_vocab = {}
        self.character_vocab = {}
        self.inverse_character_vocab = {}

        # Maps
        self.character_edges = []
        self.token_edges = []

        # Bipartite relationship
        self.character_token_nodes = {}
        self.token_entity_nodes = {}

        self.character_token_edges = []
        self.token_entity_edges = []

        # Load data in memory
        self.create_character_map()
        self.create_token_map()
        self.create_bipartite_relationships()

    def encode_characters(self, character_sequence):
        ascii_sequence = []
        for c in character_sequence:
            ascii_sequence.append(str(ord(c)))
        return ascii_sequence

    def encode_token(self, token):
        ascii_code = []
        for c in token:
            ascii_code.append(str(ord(c)))
        ascii_code = "".join(ascii_code)
        return ascii_code

    def encode_tokens(self, token_squence):
        final_converted_list = []
        for t in token_squence:
            ascii_sequence = self.encode_characters(list(t))
            final_converted_list.append("".join(ascii_sequence))
        return final_converted_list

    def create_character_map(self):
        with open(self.input_file_name, "r", encoding=self.encoding) as input_file:
            next(input_file)  # The file has a header
            edges = []
            unique_character_vocab = []
            for document in input_file:
                first_delimiter = document.find(self.delimiter)
                main_body = document[first_delimiter + 1:]
                attributes = main_body.split(self.delimiter)
                final_character_walk = []
                if " " in attributes:
                    attributes.remove(" ")
                if "" in attributes:
                    attributes.remove("")
                final_character_walk.append(chr(27))
                for attribute in attributes:
                    tokens = re.sub(" +", " ", attribute.rstrip().lstrip()).lower().split(" ")
                    for token in tokens:
                        for c in token:
                            final_character_walk.append(c)
                        if token != tokens[-1]:
                            final_character_walk.append(chr(32))
                    final_character_walk.append(chr(31))
                final_character_walk.append(chr(28))
                unique_character_vocab = unique_character_vocab + final_character_walk
                final_character_walk = self.encode_characters(final_character_walk)
                for i in range(len(final_character_walk)):
                    if i + 1 < len(final_character_walk):
                        c = final_character_walk[i]
                        next_c = final_character_walk[i + 1]
                    edges.append(str(c) + "|" + str(next_c))
            unique_character_vocab = list(set(unique_character_vocab))
            edge_counter = dict(Counter(edges))
            for c in unique_character_vocab:
                self.character_vocab[c] = [str(ord(c))]
            self.inverse_character_vocab = UtilityOperation.invert_dict(self.character_vocab)
            for k, v in edge_counter.items():
                ns2 = k.split("|")
                self.character_edges.append((str(ns2[0]), str(ns2[1]), float(v)))

    def create_token_map(self):
        with open(self.input_file_name, "r", encoding=self.encoding) as input_file:
            next(input_file)  # The file has a header
            edges = []
            unique_token_vocab = []
            for document in input_file:
                first_delimiter = document.find(self.delimiter)
                entity_id = document[0:first_delimiter]
                body = re.sub(' +', ' ', document[first_delimiter + 1:].strip().replace(self.delimiter, ' ')).lower()
                body = body.replace(" ", "*\t*")
                body = body.split("*")
                body.insert(0, "\t")
                body.append("\t")
                unique_token_vocab = unique_token_vocab + body
                body = self.encode_tokens(body)
                if "" in body:
                    body.remove("")
                self.entities_to_tokens[entity_id] = set(body)
                self.entity_vocab[entity_id] = ["".join(body)]
                for i in range(len(body)):
                    if i + 1 < len(body):
                        c = body[i]
                        next_c = body[i + 1]
                    edges.append(str(c) + "|" + str(next_c))
            unique_token_vocab = list(set(unique_token_vocab))
            unique_token_vocab.remove("")
            self.tokens_to_entities = UtilityOperation.invert_dict(self.entities_to_tokens)
            self.inverse_entity_vocab = UtilityOperation.invert_dict(self.entity_vocab)
            for t in unique_token_vocab:
                self.token_vocab[t] = [self.encode_token(t)]
                self.tokens_to_characters[self.token_vocab[t][0]] = [str(ord(c)) for c in list(t)]
            self.inverse_token_vocab = UtilityOperation.invert_dict(self.token_vocab)
            # self.tokens_to_characters[self.character_vocab[" "]] = [self.character_vocab[" "]]
            self.tokens_to_characters["9"] = ["32", "32"]
            self.characters_to_tokens = UtilityOperation.invert_dict(self.tokens_to_characters)
            edge_counter = dict(Counter(edges))
            for k, v in edge_counter.items():
                ns2 = k.split("|")
                self.token_edges.append((str(ns2[0]), str(ns2[1]), float(v)))

    def create_bipartite_relationships(self):
        for entity_id, tokens in self.entities_to_tokens.items():
            token_counts = dict(Counter(tokens))
            self.token_entity_nodes[entity_id] = "e"
            for token, count in token_counts.items():
                self.token_entity_edges.append((token, entity_id, float(count)))
                self.token_entity_nodes[token] = "t"
        for token, characters in self.tokens_to_characters.items():
            character_counts = dict(Counter(characters))
            self.character_token_nodes[token] = "t"
            for character, count in character_counts.items():
                self.character_token_edges.append((character, token, float(count)))
                self.character_token_nodes[character] = "c"

    # --------------------------------------------------------------------------------------
    # Parse PubMed articles and convert them to one character directed graph representation


    def convert_character_runs_to_graph(self, list_of_character_runs):
        final_character_walk = []
        if " " in attributes:
            attributes.remove(" ")
        if "" in attributes:
            attributes.remove("")
        final_character_walk.append(chr(27))
        for attribute in attributes:
            tokens = re.sub(" +", " ", attribute.rstrip().lstrip()).lower().split(" ")
            for token in tokens:
                for c in token:
                    final_character_walk.append(c)
                if token != tokens[-1]:
                    final_character_walk.append(chr(32))
            final_character_walk.append(chr(31))
        final_character_walk.append(chr(28))
        unique_character_vocab = unique_character_vocab + final_character_walk
        final_character_walk = self.encode_characters(final_character_walk)
        for i in range(len(final_character_walk)):
            if i + 1 < len(final_character_walk):
                c = final_character_walk[i]
                next_c = final_character_walk[i + 1]
            edges.append(str(c) + "|" + str(next_c))

        unique_character_vocab = list(set(unique_character_vocab))
        edge_counter = dict(Counter(edges))
        for c in unique_character_vocab:
            self.character_vocab[c] = [str(ord(c))]
        self.inverse_character_vocab = UtilityOperation.invert_dict(self.character_vocab)
        for k, v in edge_counter.items():
            ns2 = k.split("|")
            self.character_edges.append((str(ns2[0]), str(ns2[1]), float(v)))

    def merge_graphs(self, graph1, graph2):
        pass


    def text_to_map(self, text):
        text_to_sentences = text.split(".")
        print(text_to_sentences)
        for sentence in text_to_sentences:
            tokenized_sentence = nlp(sentence)
            tokenized_sentence = tokenized_sentence.text.lower()
            tokenized_sentence = tokenized_sentence.split(" ")
            for token in tokenized_sentence:
                character_run = list(token)





        # text_to_document = nlp(text)
        # print("------------")
        # for token in text_to_document:
        #     print(token)
        # print("------------")




    def query_documents(self, query):
        pass

    def documents_to_map(self, list_of_documents):
        for json_document in list_of_documents:
            document = json.loads(json_document)
            for subsection_header, subsection_text in document.items():
                print(subsection_header)
                print(subsection_text)
                print("-------------")


if __name__ == "__main__":
    m = MappER(",", "data/S1G.txt")
    list_of_documents = ['{"name": "John", "age": 30, "abstract": "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry\'s standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."}']
    for json_document in list_of_documents:
        document = json.loads(json_document)
        for subsection_header, subsection_text in document.items():
            print(subsection_header)
            print(subsection_text)
            print("-------------")


    # m.documents_to_map(list_of_documents)
    # m.text_to_map()
