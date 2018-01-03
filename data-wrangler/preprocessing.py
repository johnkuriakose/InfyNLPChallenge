#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to preprocess corpora for training
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
# 
# @example: python preprocessing.py test.raw test.corpus -psub

import gensim
import nltk.data
from nltk.corpus import stopwords
import argparse
import os
import re
import logging
import sys
import spacy
import time
    
# function replace_umlauts
# ... replaces german umlauts and sharp s in given text
# @param string  text
# @return string with replaced umlauts
def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res


def raise_invalid_tokenizer(tok):
    """
    ValueError exception for invalid tokenizer argument.
    """
    raise ValueError("Invalid tokenizer '" + tok + "' must be 'nltk' or 'spacy'")
    return None
    
    
def get_sentences(text):
    """
    Split sentences into text according to tokenizer argument.
    """
    if args.tokenizer == 'nltk':
        sentences = sentence_detector.tokenize(text)
    elif args.tokenizer == 'spacy':
        doc = nlp(text)
        sentences = doc.sents
    else:
        raise_invalid_tokenizer(args.tokenizer)

    return sentences
    
def get_words(sentence, tok, filter_punctuation, filter_stopwords, filter_digit):
    """
    Split sentence into words accordign to tokenizer argument.
    """
    if tok == 'nltk':
        words = nltk.word_tokenize(sentence)
        # filter punctuation and stopwords
        if filter_punctuation:
            punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
            punctuation = '?.!/;:()&+'
            words = [x for x in words if x not in punctuation_tokens]
            words = [re.sub('[' + punctuation + ']', '', x) for x in words]
        if filter_stopwords:
            stop_words = stopwords.words('german') if not args.umlauts else [replace_umlauts(token) for token in stopwords.words('german')]
            words = [x for x in words if x not in stop_words]
        if filter_digit:
            words = [word for word in words if not word.isdigit()]
    elif tok == 'spacy':
        words = [word for word in sentence]
        if filter_punctuation:
            words = [word for word in words if not word.is_stop]
        if filter_stopwords:
            words = [word for word in words if not word.is_punct]
        if filter_digit:
            words = [word for word in words if not word.is_digit]
        words = [word.text for word in words]
    else:
        raise_invalid_tokenizer(tok)
        
    return words

if __name__ == '__main__':
    # configuration
    parser = argparse.ArgumentParser(description='Script for preprocessing public corpora')
    parser.add_argument('raw', type=str, help='source file with raw data for corpus creation')
    parser.add_argument('target', type=str, help='target file name to store corpus in')
    parser.add_argument('tokenizer', type=str, default='nltk', help='set tokenizer and stopwords to nltk or spacy (default: nltk)')
    parser.add_argument('language', type=str, default='en', help='Language setting for spacy')
    parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation tokens')
    parser.add_argument('-s', '--stopwords', action='store_true', help='remove stop word tokens')
    parser.add_argument('-d', '--digits', action='store_true', help='remove digit tokens')
    parser.add_argument('-u', '--umlauts', action='store_true', help='replace german umlauts with their respective digraphs')
    parser.add_argument('-b', '--bigram', action='store_true', help='detect and process common bigram phrases')
    parser.add_argument('-t', '--tokenizer', action='store_true', default='nltk', help='set tokenizer and stopwords to nltk or spacy (default: nltk)')

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    start = time.time()

    if args.tokenizer == 'nltk':
        logging.info('loading nltk for tokenization and stopwords...')
        sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
        logging.info('nltk up!')
    elif args.tokenizer == 'spacy':
        logging.info('loading spacy for tokenization and stopwords...')
        nlp = spacy.load(args.language)
        logging.info('spacy up!')
    else:
        raise_invalid_tokenizer()

    # start preprocessing
    num_sentences = sum(1 for line in open(args.raw))
    # if not os.path.exists(os.path.dirname(args.target)):
        # os.makedirs(os.path.dirname(args.target))
    i = 1
    logging.info('preprocessing ' + str(num_sentences) + ' sentences')
    with open(args.raw, 'r', encoding='utf-8') as infile, open(args.target, 'wb') as output:
        for line in infile:
            # replace umlauts
            if args.umlauts:
                line = replace_umlauts(line)
            
            #Split sentences into text according to tokenizer argument.
            if args.tokenizer == 'nltk':
                sentences = sentence_detector.tokenize(line)
            elif args.tokenizer == 'spacy':
                doc = nlp(line)
                sentences = doc.sents
            else:
                raise_invalid_tokenizer(args.tokenizer)
                
            # process each sentence
            for sentence in sentences:
                # get word tokens
                words = get_words(sentence, args.tokenizer, args.punctuation, args.stopwords, args.digits)

                # strip words
                words = list(map(str.strip, words))
                # write one sentence per line in output file, if sentence has more than 1 word
                if len(words)>1:
                    output.write(' '.join(words).encode('utf-8').strip() + b'\n')
            # logging.info('preprocessing sentence ' + str(i) + ' of ' + str(num_sentences))
            i += 1
    end = time.time()
    logging.info('preprocessing of ' + str(num_sentences) + ' sentences finished in ' + str(end - start) + ' seconds!')

    # get corpus sentences
    class CorpusSentences:
        def __init__(self, filename):
            self.filename = filename
        def __iter__(self):
            for line in open(self.filename):
                yield line.split()

    if args.bigram:
        logging.info('train bigram phrase detector')
        bigram = gensim.models.Phrases(CorpusSentences(args.target))
        logging.info('transform corpus to bigram phrases')
        output = open(args.target + '.bigram', 'w')
        for tokens in bigram[CorpusSentences(args.target)]:
            output.write(' '.join(tokens).encode('utf8') + '\n')
