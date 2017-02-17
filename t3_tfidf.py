# -*- coding: utf-8 -*-

import re
import codecs

from nltk.stem.snowball import RussianStemmer, EnglishStemmer

from sklearn.feature_extraction.text import TfidfVectorizer


def check_structure(s):
    if re.search(ur'^[\dА-Яа-яA-Za-z]+$', s):
        if re.search(ur'^[А-Яа-яA-Za-z]+$', s):
            return 'WORD'
        elif re.search(ur'^\d+$', s):
            return 'NUMBER'
        else:
            return 'COMPLEX'
    else:
        return 'TRASH'


class Tokenizer(object):
    def __init__(self):
        self.cache = {}
        self.r_stemmer = RussianStemmer()
        self.e_stemmer = EnglishStemmer()

    def process_word(self, w):
        if w in self.cache:
            return self.cache[w]
        else:
            struct = check_structure(w)
            if struct == 'TRASH':
                w_proc = ''
            elif struct == 'WORD':
                if is_ascii(w):
                    w_proc = self.e_stemmer.stem(w)
                else:
                    w_proc = self.r_stemmer.stem(w)
            elif struct == 'NUMBER':
                w_proc = ''
            elif struct == 'COMPLEX':
                w_proc = w
            self.cache[w] = w_proc
            return w_proc

    def tokenize(self, text):
        text = preprosess_text(text)
        words = text.split(' ')
        tokens = []
        for w in words:
            tokens.append(self.process_word(w))
        tokens = [t for t in tokens if len(t)]
        return tokens


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def preprosess_text(t):
    t = t.lower()
    t = re.sub(r'[/+-.,;:]', ' ', t)
    return t


def get_tfidf(n_features=3000):
    # with codecs.open('../own_data/vocab_{}.txt'.format(vocab_name), 'r', 'utf-8') as f:
    #     vocab = f.read().split('\n')
    tokenizer = Tokenizer()
    return TfidfVectorizer(tokenizer=tokenizer.tokenize,
                           max_features=n_features)
    # return TfidfVectorizer(max_features=n_features)
