# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/4/18
# @Function : read embedding representation from Bert model output

import json
import numpy as np
import load
#from load import loadPrepareData

embedding_file = "output.jsonl"
# embedding_file2 = "output4.jsonl"
# embedding_file3 = "output5.jsonl"

# 获取Bert模型抽取的embeeding
def read_embedding(file):
    embedding = {}
    with open(file, 'r') as f:
        for row, line in enumerate(f.readlines()):
            dict = json.loads(line)
            embedding[row] = dict
    return embedding

class json_sentence():
    def __init__(self, obj):
        self.line_index = int(obj["linex_index"])
        self.features = obj["features"]
        self.tokens = [d["token"] for d in self.features[1:-1]]
        self.representations = []
        for feature in self.features[1:-1]:
            one_token_repre = []
            for i, layer in enumerate(feature["layers"]):
                if i == 0:
                    one_token_repre.append(layer["values"])
                # one_token_repre[i] = np.asarray(list(layer["values"])).astype(np.float)
            one_token_repre = np.asarray(one_token_repre)
            self.representations.append(one_token_repre)
        assert len(self.tokens) == len(self.representations)

    def get_embedding(self, orig_tokens):
        orig_to_tok_map = []
        count = 0
        new_word = []
        for i, tok in enumerate(self.tokens):
            if tok[:2] == "##":
                tok = tok[2:]
                assert "##" not in tok
            new_word.append(tok)
            if ''.join(new_word) == orig_tokens[count].lower():
                orig_to_tok_map.append(i)
                count += 1
                new_word = []
        assert len(orig_to_tok_map) == len(orig_tokens) and count == len(orig_tokens)
        orig_to_tok_map = [0] + orig_to_tok_map
        results = []
        for i, index in enumerate(orig_to_tok_map):
            if i > 0:
                if index == orig_to_tok_map[i - 1]:
                    results.append([index])
                else:
                    results.append(range(orig_to_tok_map[i - 1] + 1, index + 1))
        final_representations = []
        for indexes in results:
            final_rep = sum([self.representations[index] for index in indexes]) / len(indexes)
            final_representations.append(final_rep)
        final_representations = np.asarray(final_representations)
        final_representations = final_representations.squeeze(1)
        final_representations = final_representations.tolist()
        # final_representations.resize(final_representations.shape[1], final_representations.shape[0], final_representations.shape[2])
        # final_representations = final_representations.transpose(1, 0, 2)
        return " ".join(orig_tokens), final_representations

embedding = read_embedding(embedding_file)
# embedding2 = read_embedding(embedding_file2)
# embedding3 = read_embedding(embedding_file3)

if __name__ == "__main__":
    corpus = "data/greeting.txt"
    voc, pairs = load.loadPrepareData(corpus)
    print(pairs[0])

    error_line = []
    for pair in pairs:
        input, output = pair[0], pair[1]
        input_row, input_line = input[0], [word for word in input[1].split(" ") if word != ""]
        output_row, output_line = output[0], [word for word in output[1].split(" ") if word != ""]

        input_embedding = json_sentence(embedding[input_row])
        output_embedding = json_sentence(embedding[output_row])

        # input_representation = input_embedding.get_embedding(input_line)
        # output_representation = input_embedding.get_embedding(output_line)
        # print(input_representation)
        try:
            input_representation = input_embedding.get_embedding(input_line)
        except Exception as e:
            print(input_row)
            print(input_line)
            print(input_embedding.tokens)
            print(e)
            error_line.append(input_line)

        try:
            output_representation = output_embedding.get_embedding(output_line)
        except Exception as e:
            print(output_row)
            print(output_line)
            print(output_embedding.tokens)
            print(e)
            error_line.append(output_line)
    print(len(error_line))

    # sen_embedding = json_sentence(embedding[0])
    # print(sen_embedding.tokens)
    # # sen_embedding2 = json_sentence(embedding[1])
    # # print(sen_embedding2.tokens)
    #
    # orig_tokens = ['top', 'of', 'the', 'morning', 'to', 'you']
    # ori_sen, final_representation = sen_embedding.get_embedding(orig_tokens)
    # print(final_representation)
    # print(len(final_representation))
    # print(len(final_representation[0]))
    #
    # sen_embedding3 = json_sentence(embedding2[0])
    # print(sen_embedding3.tokens)
    # orig_tokens3 = ['top', 'of', 'morning', 'to', 'you']
    # ori_sen3, final_representation3 = sen_embedding3.get_embedding(orig_tokens3)
    # print(final_representation3)
    #
    # sen_embedding4 = json_sentence(embedding3[0])
    # print(sen_embedding4.tokens)
    # orig_tokens4 = ['top', 'of', 'morning', 'to', 'you', 'thank', 'you', 'kindly']
    # ori_sen4, final_representation4 = sen_embedding4.get_embedding(orig_tokens4)
    # print(final_representation4)

    # sen_embedding = json_sentence(embedding[809])
    # print(sen_embedding.tokens)
    #
    # orig_token = ['sos', 'sure', 'ask', 'away', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'eos']
    # ori_sen, final_representation = sen_embedding.get_embedding(orig_token)
    # print(ori_sen)
    # print(final_representation)
    # print(final_representation.shape)