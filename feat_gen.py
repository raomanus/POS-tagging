#!/bin/python
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
import pandas as pd

# Set of commonly used abbreviations.
abbreviations = {'imho', 'potd', 'l8', 'ffs', 'ianal', 'wcw', 'irl', 'nsfl', 'hmu', 'idk', 'cc', 'dgaf', 'omg', 'w/',\
				 'mtfbwy', 'omw', 'btw', 'eml', 'b4', 'nsfw', 'fath', 'tldr', 'ysk', 'bff', 'futab', 'b/c', 'jk', 'nm',\
				 'bae', 'fomo', 'qotd', 'ppl', 'eli5', 'mt', 'lol', 'fyi', 'tbh', 'lolz', 'gg', 'til', 'ymmv', 'op', 'dftba',\
				 'ftfy', 'orly', 'imo', 'wotd', 'ootd', 'em', 'oh', 'lms', 'wdymbt', 'g2g', 'ttys', 'gtg', 'sfw', 'fbf',\
				 'brb', 'idc', 'tl;dr', 'oan', 'hmb', 'otp', 'lmk', 'bc', 'mcm', 'yt', 'ttyn', 'rofl', 'btaim', 'afaik', 'f2f',\
				 'thx', 'ikr', 'hbd', 'tmi', 'txt', 'ttyl', 'gtr', 'hth', 'ama', 'nvm', 'lmao', 'ily', 'asl', 'ianad', 'wbu', 'fbo',\
				 'tbt', 'mm', 'tgif', 'tx', 'smh', 'icymi', 'gr8', 'yolo', 'dae', 'roflmao'}

# Dictionary of common contractions and their corresponding expansions.
contractions = {
"n't": 'not',
"'t": 'not', 
"'cause": 'because', 
"'ve": 'have', 
"'d": 'had', 
"'ll": 'will', 
"'s": 'is', 
"'m": 'am', 
"'am": 'madam', 
"'clock": 'of the clock', 
"'n": 'shall not', 
"'re": 'are', 
"'all": 'all'
}

slang_dict = dict()
with open("slang_dict.csv", "r") as file:
    for line in file:
        key, value = line.split(",")
        slang_dict[key] = value.strip()

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.
    Note that you can also call token2features here to aggregate feature counts, etc.
    """    
    slang_dict = dict()
    with open("slang_dict.csv", "r") as file:
        for line in file:
            key, value = line.split(",")
            slang_dict[key] = value.strip()

    for sent in train_sents:
        for i in range(len(sent)):
            if sent[i] in slang_dict:
                sent[i] = slang_dict[sent[i]]
            elif sent[i] in contractions:
                sent[i] = contractions[sent[i]]        
    

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    porter = PorterStemmer();
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # Adding stemmed version of word.
    ftrs.append("STEMMED=" + porter.stem(word))
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    # Additional features
    if word.startswith("http") or word.endswith(".com"):
        ftrs.append("IS_URL")
    if word in abbreviations:
        ftrs.append("IS_ABRV")
    if word.startswith("#"):
        ftrs.append("IS_HASHTAG")
    if word.startswith("@"):
        ftrs.append("IS_MENTION")
    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)