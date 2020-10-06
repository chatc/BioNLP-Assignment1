from fuzzywuzzy import fuzz
import string
import re
import nltk

text = '... about 10 days later i experienced headaches and nausea, but no fever or cough. I almost collapsed from ' \
       'being unable to breathe. im used to running several miles at a time with no issues. i went to urgent care and ' \
       'they gave me an emergency covid test with expedited results. 3 days later i tested positive ...'

print(text)

#load the small dyspnea lexicon
infile = open('./dyspnea_variants')
expressions = []
for line in infile:
    items = line.split('\t')
    print (items)
    expressions.append(string.strip(items[-1]))

print (expressions)

#would regular expression work?
for exp in expressions:
    if re.search(exp,text):
        print ('found!! -> ', exp)

import itertools
def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

def match_dict_similarity(text, expressions):
    '''
    :param text:
    :param expressions:
    :return:
    '''
    threshold = 0.75
    max_similarity_obtained = -1
    best_match = ''
    #go through each expression
    for exp in expressions:
        #create the window size equal to the number of word in the expression in the lexicon
        size_of_window = len(exp.split())
        tokenized_text = list(nltk.word_tokenize(text))
        for window in run_sliding_window_through_text(tokenized_text, size_of_window):
            window_string = ' '.join(window)


            similarity_score = fuzz.ratio(window_string, exp)

            if similarity_score >= threshold:
                print (similarity_score,'\t', exp,'\t', window_string)
                if similarity_score>max_similarity_obtained:
                    max_similarity_obtained = similarity_score
                    best_match = window_string
    print (best_match,max_similarity_obtained)

match_dict_similarity(text,expressions)