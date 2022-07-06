# %% [markdown]
# # Overview
# 
# This project represents an attempt to utilize machine learning and natural language processing techniques to predict NPS sentiments (Promotor, Passive, Detractor) based on surveys submitted through the NPS system. 
# 
# Author: Eric G. Suchanek, PhD for BestBuy.
# 
# 
# Methods tried: 
# * Sentiment analysis using textblob
# * RNN
# * LSTM
# * BERT
# * BoW
# 
# The following directory structure must be maintained:  
# * main directory/
# * notebook/  <-- where this notebook resides
# --data/
# -- raw/
# -- clean/
# -- pass/
# -- prom/
# -- det/
# 
# Based on: https://erleem.medium.com/nlp-complete-sentiment-analysis-on-amazon-reviews-374e4fea9976
# 
# (c) 2022 BestBuy, all rights reserved. Confidential. Do not share.

# %% [markdown]
# # Imports

# %%
# Load the TensorBoard notebook extension
%load_ext tensorboard

# %%
import re
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
import pickle

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

import datetime
from bby import bby


# %%
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

def remove_stopwords(ls):
    # Lemmatises, then removes stop words
    lemmatiser = WordNetLemmatizer()
    
    stop_english = Counter(stopwords.words()) #Here we use a Counter dictionary on the cached
                                            # list of stop words for a huge speed-up
    result = ls.translate(str.maketrans('', '', string.punctuation)) #removes [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]:

    word_tokens = word_tokenize(result)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_english]

    ls = [lemmatiser.lemmatize(word, get_wordnet_pos(word)) for word in filtered_sentence]

    #Joins the words back into a single string
    ls = " ".join(ls)
    return ls

# %% [markdown]
# # Data importing and exploration

# %%
NPS_df = pd.read_csv('../data/clean/NPS_District_125.csv')
NPS_df.head(5)

# %%
#Let's get the dataset lenght
len(NPS_df)

# %%
#How's distributed the dataset? Is it biased?
NPS_df.groupby('NPS® Breakdown').nunique()

# %%
#Checking balance of target classes
sentiments = list(NPS_df["NPS® Breakdown"].unique())

sentiment_nums = [len(NPS_df[NPS_df["NPS® Breakdown"] == sentiment]) / len(NPS_df) for sentiment in sentiments]

plt.bar(sentiments, sentiment_nums)

# %% [markdown]
# # Data cleaning
# 

# %%
#Let's keep only the columns that we're going to use
# since we have many more promoters than either passives or detractors
# we bias toward promoters. Let's equalize the distribution by grabbing the same number of promoters
# as passives or detractors.
# 
# cleaned aggregated data resulting from CleanNPS procedure/process

NPS_df = pd.read_csv('../data/clean/NPS_District_125.csv')
NPS_df = NPS_df[['Location', 'Workforce', 'NPS® Breakdown', 'NPS_Code', 'NPSCommentCleaned', 'OverallCommentCleaned']]

prom_list_mask = NPS_df['NPS_Code'] == 2
pass_list_mask = NPS_df['NPS_Code'] == 1
det_list_mask = NPS_df['NPS_Code'] == 0

prom_list = NPS_df[prom_list_mask]
pass_list = NPS_df[pass_list_mask]
det_list = NPS_df[det_list_mask]

prom_list_len = prom_list.shape[0]
pass_list_len = pass_list.shape[0]
det_list_len = det_list.shape[0]

print (f'Promoters: {prom_list_len}, Passives: {pass_list_len}, Detractors: {det_list_len}')
NPS_df = pd.DataFrame()

# since we normally have many more promoters than passive/detractors we should normalize our distribution
# sample an appropriate number of promoters to equalize distribution

prom_sample_size = (pass_list_len + det_list_len) // 2
prom_list = prom_list.sample(prom_sample_size)

NPS_df = prom_list.copy()
NPS_df = NPS_df.append(pass_list, ignore_index=True)
NPS_df = NPS_df.append(det_list, ignore_index=True)

#Checking balance of target classes
sentiments = list(NPS_df["NPS® Breakdown"].unique())
sentiment_nums = [len(NPS_df[NPS_df["NPS® Breakdown"] == sentiment]) / len(NPS_df) for sentiment in sentiments]

print (f'After redistribution: Promoters: {prom_list.shape[0]}, Passives: {pass_list_len}, Detractors: {det_list_len}')
plt.bar(sentiments, sentiment_nums)

# %%
# grab the NPS and NPS overall comments for all stores
overall_list = NPS_df['OverallCommentCleaned']
nps_list = NPS_df['NPSCommentCleaned']

NPS_df['OverallCommentLemmatised'] = overall_list.apply(remove_stopwords)
NPS_df['NPSCommentLemmatised'] = nps_list.apply(remove_stopwords)

overall_list = NPS_df['OverallCommentLemmatised']
nps_list = NPS_df['NPSCommentLemmatised']

NPS_df.to_csv('../data/clean/NPS_District_125_subset.csv', index=False)



# %%
from keras.preprocessing.text import text_to_word_sequence
def sent_to_words(sentences):
    for sentence in sentences:
        yield(text_to_word_sequence(sentence))
        
# grab the lemmatised overall NPS comments
data_words = list(sent_to_words(nps_list))

# print(data_words[:10])

# %%
len(data_words)

# %%
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

# %%
data = []

for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
print(data[:5])

# %%
data = np.array(data)

# %% [markdown]
# # Label encoding
# 
# As the dataset is categorical, we need to convert the sentiment labels from Detractor, Passive and Promoter to a float type that our model can understand. To achieve this task, we'll implement the to_categorical method from Keras.

# %%
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
NPS_df['code'] = LE.fit_transform(NPS_df['NPS® Breakdown'])
NPS_df.head()

# %%
labels = np.array(NPS_df['NPS® Breakdown'])
y = []
for i in labels:
    if i == 'Promoter':
        y.append(2.0)
    elif i == 'Passive':
        y.append(1.0)
    elif i == 'Detractor':
        y.append(0.0)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
del y
print(len(labels))

# %%
# Utility function: Define the indexing for each possible label in a dictionary

class_to_index = {"Detractor":0, "Passive":1, "Promoter":2, }

#Creates a reverse dictionary
index_to_class = dict((v,k) for k, v in class_to_index.items())

#Creates lambda functions, applying the appropriate dictionary
names_to_ids = lambda n: np.array([class_to_index.get(x) for x in n])
ids_to_names = lambda n: np.array([index_to_class.get(x) for x in n])

# %% [markdown]
# # Data sequencing and splitting
# 
# We'll implement the Keras tokenizer as well as its pad_sequences method to transform our text data into 3D float data, otherwise our neural networks won't be able to be trained on it.

# %%
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
max_words = 1000
# was 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
comments = pad_sequences(sequences, maxlen=max_len)

# %%
#
#  use scikit-multilearn since we are using a non-binary (trinary output)
import skmultilearn
from skmultilearn.model_selection import iterative_train_test_split
X_train, y_train, X_test, y_test = iterative_train_test_split(comments, labels, test_size = 0.3)


# %%
from keras.backend import clear_session
# Before instantiating a tf.data.Dataset obj & before model creation, call:
clear_session()

# %% [markdown]
# # Model building
# 
# In the next cells we experiment with several different Neural Networks. I'll implement sequential models from the Keras API to achieve this task. Essentially, I'll start with a single layer **LSTM** network which is known by achieving good results in NLP tasks when the dataset is relatively small (I could have started with a SimpleRNN which is even simpler, but to be honest it's actually not deployed in production environments because it is too simple - however I'll leave it commented in case you want to know it's built). The next one will be a Bidirectional LSTM model, a more complex one and this particular one is known to achieve great metrics when talking about text classification. To go beyond the classic NLP approach, finally we'll implement a very unusual model: a Convolutional 1D network, known as well by delivering good metrics when talking about NLP. If everything goes ok, we should get the best results with the BidRNN, let's see what happens.
# 

# %% [markdown]
# ## SimpleRNN model (Bonus)

# %%
model0 = Sequential()
model0.add(layers.Embedding(max_words, 15))
model0.add(layers.SimpleRNN(15))
model0.add(layers.Dense(3,activation='softmax'))

log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

metric = 'val_accuracy'
model0.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint0 = ModelCheckpoint("best_model0.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1, save_weights_only=False)
history = model0.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[checkpoint0, tensorboard_callback])

# %%
# plot learning curves
import matplotlib.pyplot as pyplot

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

# %% [markdown]
# ## Single LSTM layer model

# %%
model1 = Sequential()
model1.add(layers.Embedding(max_words, 20))
model1.add(layers.LSTM(15,dropout=0.5))
model1.add(layers.Dense(5, activation='tanh'))
model1.add(layers.Dense(3,activation='softmax'))

log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='loss', verbose=0, save_best_only=True, mode='auto', save_freq=1,save_weights_only=False)
history = model1.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test),callbacks=[checkpoint1, tensorboard_callback])

# %%
model1b = Sequential()
model1b.add(layers.Embedding(max_words, 20))
model1b.add(layers.LSTM(15,dropout=0.5))
model1b.add(layers.Dense(10,activation='tanh'))
model1b.add(layers.Dense(3,activation='softmax'))

log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model1b.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint("best_model4.hdf5", monitor='loss', verbose=0, save_best_only=True, mode='auto', save_freq=1,save_weights_only=False)
history = model1b.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint1, tensorboard_callback])

# %%
# %tensorboard --logdir ../logs/fit

# %%
# plot learning curves
import matplotlib.pyplot as pyplot

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

# %% [markdown]
# ## Bidirectional LTSM model

# %%
log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model2.add(layers.Dense(3,activation='softmax'))
model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1,save_weights_only=False)
history = model2.fit(X_train, y_train, epochs=75, validation_data=(X_test, y_test), callbacks=[checkpoint2, tensorboard_callback])

# %%
#%tensorboard --logdir logs/fit

# %%
# plot learning curves

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

# %% [markdown]
# ## 1D Convolutional model
# 
# Before diving into this model, I know by prior experience that it tends to overfit extremely fast on small datasets. In this sense, just will implement it to show you how to do it in case it's of your interest.

# %%
# 
from keras import regularizers
log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model3 = Sequential()
model3.add(layers.Embedding(max_words, 40, input_length=max_len))
model3.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.MaxPooling1D(5))
model3.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.GlobalMaxPooling1D())
model3.add(layers.Dense(3,activation='softmax'))
model3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint3 = ModelCheckpoint("best_model3.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1,save_weights_only=False)
history = model3.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=[checkpoint3, tensorboard_callback])

# %%
# plot learning curves
import matplotlib.pyplot as pyplot

pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Cross Entropy')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()

# %% [markdown]
# If you check the val_accuracy metric in the training logs you won't find better score than the one achieved by the BidRNN. Again, the previous model is not the best for this task becaue is majorly used for short translation tasks, but the good thing to notice is its speed to train.
# 
# Let's move on.

# %% [markdown]
# # Best model validation
# (Before final commit, the best model obtained was the BidRNN)

# %%
#Let's load the best model obtained during training
best_model = keras.models.load_model("best_model2.hdf5")

# %%
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ',test_acc)

# %%
predictions = best_model.predict(X_test)

# %% [markdown]
# ## Confusion matrix
# 
# Alright, we all know the accuracy is not a good metric to measure how well a model is. That's the reason why I like to always see its confusion matrix, that way I have a better understanding of its classification and generalization ability. Let's plot it.

# %%
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))

# %%
import seaborn as sns
conf_matrix = pd.DataFrame(matrix, index = ['Detractor','Passive','Promoter'],columns = ['Detractor','Passive','Promoter'])
#Normalizing
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (15,15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})

# %% [markdown]
# Again, the model's score is very poor, but keep in mind it hasn't gone through hyperparameter tuning. Let's see how it performs on some test text.

# %%
sentiment = ['Detractor','Passive','Promoter']

# %%
sequence = tokenizer.texts_to_sequences(['this experience has been the worst , want my money back'])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]

# %%
sequence = tokenizer.texts_to_sequences(['Eric Suchanek is the best agent ever!'])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]

# %%
sequence = tokenizer.texts_to_sequences(['i hate youtube ads, they are annoying'])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]

# %%
sequence = tokenizer.texts_to_sequences(['i really loved how the technician helped me with the issue that i had'])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]

# %% [markdown]
# We've reached the end of this notebook. I just wanted to highlight a few things before let you go.
# 
# As you could see, very simple networks can achieve fantastic results. To go beyond, always the best approach is to build a model that underfit the data, then optimize it to overfit and finally start tuning your hyperparameters to achieve the metric that the business needs to reach. The way you tune the model is up to you, there's no magic formula for it, but adding regularization always works, as well as dropout. 
# 
# If you have any doubt, please feel free to comment :)

# %% [markdown]
# 


