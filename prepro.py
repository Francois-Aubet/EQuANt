import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import random


'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


""" The prepocessing file, does it according to the paper.

The 'main' is the last method of the file.

"""

# Create a blank model of a given language class
nlp = spacy.blank("en")


def word_tokenize(sent):
    # creates a list of words from the 'sent' string
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    """ Creates the list of spans from the answer and the context. """
    """ This is a thing we have to addapt for the squad 2.0. """
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans



def process_file(filename, data_type, word_counter, char_counter):
    """ Processes a file to obtain the context, answers, and asnwers spans sepeartely. """

    # a boolean to say if we are creating a very small dataset that can be used for debugging
    small_dataset = False
    shuffled_dataset = False


    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    # total number of questions:
    total = 0
    # read from source file using the jason library:
    with open(filename, "r") as fh:
        source = json.load(fh)

        source_data = source["data"]

        # downsample the dataset:
        #source_data = random.sample(source_data, int(float(len(source_data))*(1-0.9)))
        
        # for each paragraph in each article:
        for index, article in enumerate(tqdm(source_data)):
            if small_dataset:
                article = source_data[0]
                num_para = 0

            if shuffled_dataset:
                if index+1 > len(source_data):
                    article_for_ct = source_data[0]
                else:
                    article_for_ct = source_data[index+1]
            

            for para in article["paragraphs"]:
                if small_dataset:
                    if num_para > 3:
                        break
                    else:
                        num_para += 1

                # replacing characters that should be the same:
                context = para["context"].replace("''", '" ').replace("``", '" ')


                if shuffled_dataset:
                    context = article_for_ct["paragraphs"][0]["context"].replace("''", '" ').replace("``", '" ')

                # tokenize the context and find the answer spans:
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                # we count the occurances of each word and char
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])


                if len(context_tokens) < 33:
                    print(len(context_tokens))
                    print(context)

                # for each question of the paragraph:
                for qa in para["qas"]:
                    total += 1

                    # replacing characters that should be the same:
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')

                    # tockenize the words of the question and cont them:
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    
                    # findout if the question has an answer:
                    has_answer = 1.0
                    if not qa["answers"]:
                        has_answer = 0.0

                    # get the answer start and end indices (if there are answers):
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)

                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "y3": has_answer, "id": total}
                    #if has_answer == 0.0:
                    examples.append(example)
                    eval_examples[str(total)] = { "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}

            if small_dataset:
                break

        # shuffle examples for the training:               
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples



def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    """
    Get the embeding of each word/character from the the embedding file.
    """
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    # get the words present in the dataset:
    filtered_elements = [k for k, v in counter.items() if v > limit]

    if emb_file is not None:
        assert size is not None
        assert vec_size is not None

        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1

    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    return emb_mat, token2idx_dict


def _get_word(word,word2idx_dict,is_test):
    """ Returns the embedding of the word in the created embedding dictionary
        If the word is not found, returns 1.
    """
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2idx_dict:
            return word2idx_dict[each]
    #if not is_test:
        #print(word + " not contained..")
    return 1

def _get_char(char,char2idx_dict,is_test):
    """ Returns the embedding of the char in the created embedding dictionary
        If the char is not found, returns 1.
    """
    if char in char2idx_dict:
        return char2idx_dict[char]
    #if not is_test:
        #print(char + " not contained..")
    return 1


def build_features(examples, data_type, out_file, word2idx_dict, char2idx_dict, para_limit, ques_limit, ans_limit, char_limit, is_test=False):
    """ 
    Takes the list of examples as input and transforms them to there embedded version.

    """

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               ( (example["y3"] == 1) and (example["y2s"][0] - example["y1s"][0]) > ans_limit)

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    # true total of examples
    total = 0
    # total of examples tried:
    total_ = 0
    meta = {}

    number_unknwonwords = 0
    number_known_words = 0

    # for each example in the dataset:
    for example in tqdm(examples):
        total_ += 1

        # if one of the criteria is not respected, we discard this example
        if filter_func(example, is_test):
            continue

        total += 1

        # the encoding of the context's words:
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        # the encoding of the context's chars:
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        # the encoding of the question's words:
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        # the encoding of the question's chars:
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        # the one hot encodings of the start and end positions: 
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)
        y3 = np.zeros(1, dtype=np.float32)


        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token,word2idx_dict,is_test)
            if context_idxs[i] == 1:
                number_unknwonwords += 1
            else:
                number_known_words += 1

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token,word2idx_dict,is_test)
            if context_idxs[i] == 1:
                number_unknwonwords += 1
            else:
                number_known_words += 1

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char,char2idx_dict,is_test)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char,char2idx_dict,is_test)

        if (example["y3"] == 1):   
            start, end = example["y1s"][-1], example["y2s"][-1]
            y1[start], y2[end] = 1.0, 1.0

        y3[0] = example["y3"]

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "y3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y3.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())

    print("In total: " + str(number_unknwonwords) + " unkown words ... .. .")
    print("In total: " + str(number_known_words) + " kown words!")
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()

    return meta



def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)



def prepro(config):
    """ The 'main' of preprocessing that is called to do it and calls the other functions.
    
    :param: config : the flags object
    :return: nothing, only saves the created datasets in the specified files
    """

    # preprocess all the datasets and count the number of occurances of each word and char:
    word_counter, char_counter = Counter(), Counter()

    train_examples, train_eval = process_file(config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(config.dev_file, "dev", word_counter, char_counter)    
    test_examples, test_eval = process_file(config.test_file, "test", word_counter, char_counter)


    # read the files from the config:
    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    if not config.use_squad1_dict:
        # get the embedding for the words and characters in the files:
        word_emb_mat, word2idx_dict = get_embedding(
            word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
        char_emb_mat, char2idx_dict = get_embedding(
            char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)
    else: 
        # we just load the emb dictionaries and dictionaries from squad1
        with open(config.word_dictionary, "r") as fh:
            word2idx_dict = json.load(fh)
        with open(config.char_dictionary, "r") as fh:
            char2idx_dict = json.load(fh)

    # build the features of each of the datasets.
    build_features(train_examples, "train",config.train_record_file, word2idx_dict, char2idx_dict,
    config.para_limit,config.ques_limit,config.ans_limit,config.char_limit)

    dev_meta = build_features(dev_examples, "dev",config.dev_record_file, word2idx_dict, char2idx_dict,
    config.para_limit,config.ques_limit,config.ans_limit,config.char_limit)

    test_meta = build_features(test_examples, "test",config.test_record_file, word2idx_dict, char2idx_dict,
     config.test_para_limit,config.test_ques_limit,100,config.char_limit, is_test=True)

    # saving everything
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.test_meta, test_meta, message="test meta")

    if not config.use_squad1_dict:
        save(config.word_emb_file, word_emb_mat, message="word embedding")
        save(config.char_emb_file, char_emb_mat, message="char embedding")
        save(config.word_dictionary, word2idx_dict, message="word dictionary")
        save(config.char_dictionary, char2idx_dict, message="char dictionary")






    # downsample the dataset:
    # train_examples = random.sample(train_examples, int(float(len(train_examples))*(1-config.portion_downsampling)))
    # dev_examples = random.sample(dev_examples, int(float(len(dev_examples))*(1-config.portion_downsampling)))

# para_limit = config.test_para_limit if is_test else config.para_limit
# ques_limit = config.test_ques_limit if is_test else config.ques_limit
# ans_limit = 100 if is_test else config.ans_limit
# char_limit = config.char_limit





# def convert_to_features(config, data, word2idx_dict, char2idx_dict):

#     example = {}
#     context, question = data
#     context = context.replace("''", '" ').replace("``", '" ')
#     question = question.replace("''", '" ').replace("``", '" ')
#     example['context_tokens'] = word_tokenize(context)
#     example['ques_tokens'] = word_tokenize(question)
#     example['context_chars'] = [list(token) for token in example['context_tokens']]
#     example['ques_chars'] = [list(token) for token in example['ques_tokens']]

#     para_limit = config.test_para_limit
#     ques_limit = config.test_ques_limit
#     ans_limit = 100
#     char_limit = config.char_limit

#     def filter_func(example):
#         return len(example["context_tokens"]) > para_limit or \
#                len(example["ques_tokens"]) > ques_limit

#     if filter_func(example):
#         raise ValueError("Context/Questions lengths are over the limit")

#     context_idxs = np.zeros([para_limit], dtype=np.int32)
#     context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
#     ques_idxs = np.zeros([ques_limit], dtype=np.int32)
#     ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
#     y1 = np.zeros([para_limit], dtype=np.float32)
#     y2 = np.zeros([para_limit], dtype=np.float32)

#     def _get_word(word):
#         for each in (word, word.lower(), word.capitalize(), word.upper()):
#             if each in word2idx_dict:
#                 return word2idx_dict[each]
#         return 1

#     def _get_char(char):
#         if char in char2idx_dict:
#             return char2idx_dict[char]
#         return 1

#     for i, token in enumerate(example["context_tokens"]):
#         context_idxs[i] = _get_word(token)

#     for i, token in enumerate(example["ques_tokens"]):
#         ques_idxs[i] = _get_word(token)

#     for i, token in enumerate(example["context_chars"]):
#         for j, char in enumerate(token):
#             if j == char_limit:
#                 break
#             context_char_idxs[i, j] = _get_char(char)

#     for i, token in enumerate(example["ques_chars"]):
#         for j, char in enumerate(token):
#             if j == char_limit:
#                 break
#             ques_char_idxs[i, j] = _get_char(char)

#     return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs