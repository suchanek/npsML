# Include file for Best Buy machine learning project.
# Copyright (c) 2022, Best Buy Inc, all rights reserved
# Author: Eric G. Suchanek, PhD.
# Best Buy Confidential, do not distribute.
#
# We are in territory 33, market 21, district 125, store 0494
#
import string
import numpy as np
import datetime
import re

import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from collections import Counter

import shutil
import glob, os, os.path

#TextBlob Features
import textblob
from textblob import TextBlob

DEBUG = False

# Globals for our store
Our_Territory = 3
Our_Market = 21
Our_District = 125
Our_Store = "00494"

Territory_list = [1, 2, 3, 4, 14]

# List of all markets within territory 3
market_list_3 = [4, 5, 21, 24, 78, 79, 94]
market_list_locale_dict = {4:"Detroit/Indy", 5:"Boston", 21:"Ohio/Pittsburgh", 24:"Philly/Upstate NY", 78:"MKT 78 Philadelphia", 
                    79:"mkt 79 New York City", 93:"T33 Pooled Labor"}

# a list containing district numbers for each market within territory 3
district_list_4 = [16, 17, 49]
district_list_5 = [37, 39, 107]
district_list_21 = [14, 23, 52, 93, 125, 127]
district_list_24 = [43, 54, 87, 126]
district_list_78 = [235, 525, 530, 532, 533, 537, 538, 539, 540, 6541, 548]
district_list_79 = [215, 216, 217, 218, 219, 221, 222, 223, 224, 25, 226, 227, 228, 229, 231, 232, 233, 234, 252]
district_list_94 = [214, 220, 230, 479, 480, 484, 492, 498]

# Store numbers for each District within market 21 - we use text fields to keep the formatting right
district_stores_14 = [ "00228", "00229", "00368", "00371", "00490", "00491", "01025", "01094"]
district_stores_23 = [ "00162", "00168", "00271", "00278", "00279", "00285", "00286", "00758", "00879", "00880", "01050", "01099"]
district_stores_52 = [ "00227", "00259", "00333", "00335", "00791", "01010", "01121", "01477"]
district_stores_93 = [ "00145", "00156", "00292", "00295", "00339", "00390", "00570", "00573", "01096", "01266"]
district_stores_125 = [ "00154", "00161", "00266", "00274", "00617", "00494", "00790", "01252", "01474"]
district_stores_127 = [ "00230", "00232", "00327", "00489", "00858", "02501", "02512"]

# dict for all stores within market 21
district_stores_21_dict = {14:district_stores_14, 125:district_stores_23, 
                           52:district_stores_52, 93:district_stores_93, 125:district_stores_125,
                           127:district_stores_127}

district_stores_dict = {21:district_stores_21_dict,}

# global variables used for pathing relative to the notebook directory
#
_txt_extension = ".txt"
_raw_path = "../data/raw/"
_cleaned_path = "../data/clean/"    
_filename_prefix = "export_Main Hierarchy_"
_output_filename_prefix = "NPS_cleaned_"
_output_filename_prefix_natl = "NPS_NATL_"

_prom_words_path = _cleaned_path + "prom/"
_pass_words_path = _cleaned_path + "pass/"
_det_words_path = _cleaned_path + "det/" 

def clean_txt_dirs():
    filelist = glob.glob(os.path.join(_prom_words_path, "*.txt"))
    
    for f in filelist:
        os.remove(f)
    
    filelist = glob.glob(os.path.join(_pass_words_path, "*.txt"))
    for f in filelist:
        os.remove(f)

    filelist = glob.glob(os.path.join(_det_words_path, "*.txt"))
    for f in filelist:
        os.remove(f)   
    
    return

def replace_label(original_file, new_file):
  # Load the original file to pandas. We need to specify the separator as
  # '\t' as the training data is stored in TSV format
  df = pd.read_csv(original_file, sep='\t')

  # Define how we want to change the label name
  label_map = {0: 'Detractor', 1: 'Passive', 2: 'Promotor'}

  # Excute the label change
  df = df.replace({'label': label_map}, inplace=True)

  # Write the updated dataset to a new file
  df.to_csv(new_file)

from string import punctuation
import os
from os import listdir

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_train):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

#
# Return the part of speech for the givin word in the format lemmatize() accepts
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatises, then removes stop words and performs stemming and returns a string
def lemma_remove_stopwords(_ls):
    # get a Lemmatizer and Stemmer
    lemmatiser = WordNetLemmatizer()
    porter = PorterStemmer()

    ls = str(_ls)

    # build a Counter dict containing English stopwords
    stop_english = Counter(stopwords.words()) # Here we use a Counter dictionary on the cached
                                            # list of stop words for a huge speed-up
    result = ls.translate(str.maketrans('', '', string.punctuation)) #removes [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:
    word_tokens = word_tokenize(result)

    # filtered_sentence = [porter.stem(w) for w in word_tokens if not w.lower() in stop_english]
    # try without stemming
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_english and len(w) > 1]

    # lemmatize
    ls = [lemmatiser.lemmatize(word, get_wordnet_pos(word)) for word in filtered_sentence]

    #Joins the words back into a single string
    ls = " ".join(ls)
    return ls


# given an input string, remove words shorter than 1 char, return
# cleaned string
                                         
def Onps_cleanstring(comment):
    STOP_english = Counter(stopwords.words()) # Here we use a Counter dictionary on the cached
    Vocab_str = comment.lower()
    
    # print(f'Vocab_string <{Vocab_str}>')
    
    res = Vocab_str.translate(str.maketrans('', '', string.punctuation)) #removes [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:
    docWords = res.split()
    filtered_sentence = [w for w in docWords if not w.lower() in STOP_english and len(w) > 1]
    
    ls = " ".join(filtered_sentence)
    return ls

def nps_cleanstring(_string):
    doc = str(_string)
    ls = str()
    
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word.lower() for word in tokens if word.isalpha() and len(word) > 1]

    ls = TreebankWordDetokenizer().detokenize(tokens)
    return ls

# given an input string, remove stop words return cleaned string

def nps_remove_stopwords(_string):
    doc = str(_string)
    ls = str()
    GS_stopwords = ['geek', 'squad', 'service', 'customer', 'xyxyxz','\`', '\'']

    stop_words = set(stopwords.words())
    stop_words.update(GS_stopwords)
 
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word.lower() for word in tokens if not word in stop_words]

    ls = TreebankWordDetokenizer().detokenize(tokens)
    return ls

def nps_lemmatise(_ls):
    # get a Lemmatizer and Stemmer
    lemmatiser = WordNetLemmatizer()
    
    ls = str(_ls)
    tokens = word_tokenize(ls)

    # lemmatize
    ls = [lemmatiser.lemmatize(word.lower(), get_wordnet_pos(word)).lower() for word in tokens]
    ls = TreebankWordDetokenizer().detokenize(tokens)
    return ls

def sent_to_words(sentences):
    for sentence in sentences:
        yield(text_to_word_sequence(sentence))
    return

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

# Utility function: Define the indexing for each possible label in a dictionary

class_to_index = {"Detractor":0, "Passive":1, "Promoter":2, }

#Creates a reverse dictionary
index_to_class = dict((v,k) for k, v in class_to_index.items())

#Creates lambda functions, applying the appropriate dictionary
names_to_ids = lambda n: np.array([class_to_index.get(x) for x in n])
ids_to_names = lambda n: np.array([index_to_class.get(x) for x in n])

# Word frequency distributions
def nps_freqs(stringlist, howmany=0):
    Vocab_str = str()
    Vocab_str = " ".join(str(review).lower() for review in stringlist)

    docSplit = Vocab_str.split()
    freq = nltk.FreqDist(w for w in docSplit)
    return freq

def tb_enrich(ls):
    #Enriches a column of text with TextBlob Sentiment Analysis outputs
    tb_polarity = []
    tb_subject = []

    for comment in ls:
        sentiment = TextBlob(comment).sentiment
        tb_polarity.append(sentiment[0])
        tb_subject.append(sentiment[1])
    return tb_polarity, tb_subject


def remove_stopwords(sent):
    filtered_sentence = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sent)
 
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
 
    return(filtered_sentence)

def cleanup_data(data):
    
    # Remove distracting 
    data = re.sub("\`", "", data)
        
    return data

def ocleanup_data(data):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    #print(data)
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    # Remove elipsis
    data = re.sub("\...",  " ", data)

    # Strip escaped quotes
    data = re.sub('\\"', '', data)
 
    # Strip quotes
    data = re.sub('"', '', data)
    
    # Strip !
    data = re.sub('!', '', data)

    data = re.sub("\`", "", data)
        
    return data


def decontractions(phrase):
    #specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase

#
# merge all the old .xlsx files, drop duplicates, store in ../data/raw
# as an 'archive' file containing all comments -egs-
def NPS_merge_old():
    oldpath = "../data/raw/old/"
    rawpath = "../data/raw/"
    pat = oldpath + "*.xlsx"
    combinedfn = rawpath + 'NPS_NATL_archive.xlsx'
    new_df = pd.DataFrame()

    old_files = glob.glob(pat)
    tot = len(old_files)
    i = 1
    old_df = pd.DataFrame()
    old_dflist = []

    for filename in old_files:
        print(f'{i}/{tot} Reading {filename}')
        old_df = pd.read_excel(filename, header=3)
        old_df.set_index('respid2')
        old_dflist.append(old_df)
        print(f'    Read: {old_df.shape[0]} records from {filename}')
        i += 1
    
    new_df = pd.concat(old_dflist, axis=0)
    new_df.set_index('respid2')
    new_df = new_df.drop_duplicates()

    print(f'Newly combined dataset has {new_df.shape[0]} entries.')
    new_df.to_excel(combinedfn, index=False)
    print(f'Wrote file: {combinedfn} with {new_df.shape[0]} entries.')

    return

# end of file
