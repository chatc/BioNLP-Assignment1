import pandas as pd
from collections import defaultdict
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import *
from fuzzywuzzy import fuzz
import itertools
from tqdm import tqdm

stemer = PorterStemmer()

SYM_MAX_LENGTH = 5
def load_labels(f_path):
    '''
    Loads the labels

    :param f_path:
    :return:
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(list)
    for index, row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['Symptom CUIs']) and not pd.isna(row['Negation Flag']):
            cuis = row['Symptom CUIs'].split('$$$')[1:-1]
            neg_flags = row['Negation Flag'].split('$$$')[1:-1]
            for cui, neg_flag in zip(cuis, neg_flags):
                labeled_dict[id_].append(cui + '-' + str(neg_flag))
    return labeled_dict


def load_texts(f_path):
    '''
    Loads the texts

    :param f_path:
    :return dict of (id, text) pairs
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(str)
    for index, row in labeled_df.iterrows():
        id_ = row['ID']
        if not pd.isna(row['TEXT']):
            labeled_dict[id_] = row['TEXT']
    return labeled_dict


def load_symptom(path):
    inverse_dict = defaultdict(list)
    symptoms_dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split('\t')
            inverse_dict[line[-2].strip()].append(line[-1].strip().lower())
            symptoms_dict[line[-1].strip().lower()] = line[-2].strip()
    return symptoms_dict, inverse_dict


def load_my_label(path):
    my_sympton_dict = {}
    labeled_df = pd.read_excel(path)
    for index, row in labeled_df.iterrows():
        # print(row)
        if type(row['Symptom CUIs']) != str:
            continue
        cuis = row['Symptom CUIs'].split('$$$')[1:-1]
        expressions = row['Symptom Expressions'].split('$$$')[1:-1]
        for cui, exp in zip(cuis, expressions):
            my_sympton_dict[exp] = cui
    return my_sympton_dict


def load_neg(path):
    infile = open(path, 'r', encoding='utf-8')
    text = infile.read()
    negs = [x.strip() for x in text.split('\n')]
    return negs


def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words)  # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator,
                                         window_size))  # islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    # now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window


def match_dict_similarity(text, exp):
    '''
    :param text:
    :param expression:
    :return:
    '''
    threshold = 75
    max_similarity_obtained = -1
    best_match = ''
    # go through each expression
    # create the window size equal to the number of word in the expression in the lexicon
    size_of_window = len(exp.split())
    tokenized_text = list(nltk.word_tokenize(text))
    similar_words = []
    threshold = max(threshold, 100 - 5*(size_of_window-1))
    # threshold = 75
    for window in run_sliding_window_through_text(tokenized_text, size_of_window):
        window_string = ' '.join(window)
        similarity_score = fuzz.ratio(window_string, exp)

        if similarity_score >= threshold:
            # print(similarity_score, '\t', 'symtom:', exp, '\t', 'text:', window_string)
            if similarity_score > max_similarity_obtained:
                max_similarity_obtained = similarity_score
                best_match = window_string
            similar_words.append(window_string)
    return similar_words, best_match


if __name__ == '__main__':
    # load data
    texts = load_texts("./Assignment1GoldStandardSet.xlsx")
    symptom_dict, _ = load_symptom("./COVID-Twitter-Symptom-Lexicon.txt")
    dys_dict, _ = load_symptom("./dyspnea_variants")
    my_sympton_dict = load_my_label("./s2.xlsx")
    negs = load_neg('neg_trigs.txt')

    for x, y in my_sympton_dict.items():
        symptom_dict[x] = y
    for x, y in dys_dict.items():
        symptom_dict[x] = y
    sympton_list = sorted(symptom_dict.items(), key=lambda x: -len(word_tokenize(x[0])))
    sympton_list = [x for x in sympton_list if len(word_tokenize(x[0])) <= SYM_MAX_LENGTH]
    # exact match
    answers_dict = defaultdict(list)
    pd_ids = []
    pd_CUIs = []
    pd_flags = []
    tot = 0
    for id, text in tqdm(texts.items()):
        print('\n')
        output_CUI = "$$$"
        output_flag = "$$$"
        for sentence in sent_tokenize(text):
            # special '-'
            replaced_sentence = re.sub('-', lambda x: ' ', sentence.lower())
            # special '/'
            replaced_sentence = re.sub('/', lambda x: ' / ', sentence.lower())
            replaced_token = []
            for i, token in enumerate(word_tokenize(replaced_sentence)):
                if (token == '/' or token == 'or') and i != 0:
                    if replaced_token[-1] in ['smell', 'taste']:
                        repeat = replaced_token[-min(len(replaced_token), 3): -1]
                        replaced_token += repeat
                    else:
                        replaced_token.append(token)
                else:
                    replaced_token.append(token)
            replaced_sentence = ' '.join(replaced_token)

            # start matching

            for sympton, CUI in sympton_list:
                # exact match
                # replaced_sentence = re.sub(r"\b{}\b".format(sympton), lambda x: CUI, replaced_sentence)
                # fuzzy match
                similar_phrases, _ = match_dict_similarity(replaced_sentence, sympton)
                for phrase in similar_phrases:
                    try:
                        replaced_sentence = re.sub(r"\b{}\b".format(phrase), lambda x: CUI, replaced_sentence)
                    except Exception:
                        # print(phrase)
                        continue
            for neg in negs:
                replaced_sentence = re.sub(neg, lambda x: 'NEG', replaced_sentence)
            replaced_sentence = word_tokenize(replaced_sentence)
            neg_count = 0
            for token in replaced_sentence:
                if token == '.':
                    neg_count = 0
                if token == 'NEG':
                    neg_count = 4
                elif re.match(r'C\d{7}', token) != None:
                    tot += 1
                    prefix = "{}.".format(id)
                    output_CUI += token + '$$$'
                    if neg_count > 0:
                        print(prefix, token + '-1')
                        output_flag += "1$$$"
                    else:
                        print(prefix, token + '-0')
                        output_flag += "0$$$"
                if neg_count: neg_count -= 1

        pd_ids.append(id)
        pd_CUIs.append(output_CUI)
        pd_flags.append(output_flag)

    df = pd.DataFrame({'ID': pd_ids, 'Symptom CUIs': pd_CUIs, 'Negation Flag': pd_flags})
    df.to_excel('result.xlsx', index=False)
    print('total found:', tot)