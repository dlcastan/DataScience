# Importo librerías
import pandas as pd
import numpy as np
import nltk
import itertools
import pandas as pd
import pickle
import sklearn as sk
import tensorflow as tf
import keras as kr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


nltk.download('punkt')
nltk.download('stopwords')

#Importo las librerías que voy a necesitar
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#Cargo datos desde Drive
data = pd.read_json("dataset_es_dev.json", lines = True)

# Cargo las Stopwords para el idioma español
from nltk.corpus import stopwords
stopwords.words('spanish')

# Sumo palabras al stopwords
stopwords = nltk.corpus.stopwords.words('spanish')

from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

import re
from nltk.stem import PorterStemmer

dataset = data.copy()
dataset.dropna(axis=0,inplace=True)
stemmer = PorterStemmer()

reviews_list=[]
for reviews in dataset.review_body:
    reviews = re.sub("[^a-zA-Z]"," ", str(reviews)) 
    reviews = reviews.lower()
    reviews = word_tokenize(reviews)
    reviews = [stemmer.stem(palabra) for palabra in reviews]
    reviews = [palabra for palabra in reviews if len(palabra)>3 ]
    reviews = [palabra for palabra in reviews if not palabra in stopwords]
    reviews = " ".join(reviews)
    reviews_list.append(reviews)

dataset["review_stem"] = reviews_list


# Importamos esta libreria que nos permite reemplzar caracteres
wordnet_lemmatizer = WordNetLemmatizer()

reviews_list=[]
for reviews in dataset.review_body:
    reviews = re.sub("[^a-zA-Z]"," ", str(reviews)) 
    reviews = reviews.lower()
    reviews = word_tokenize(reviews)
    review_lemma = [wordnet_lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in reviews]
    reviews = [palabra for palabra in reviews if len(palabra)>3 ]
    reviews = [palabra for palabra in reviews if not palabra in stopwords]
    reviews = " ".join(reviews)
    reviews_list.append(reviews)

dataset["review_lemm"] = reviews_list

# Creo columna de opinion positiva(1) y negativa(0)
dataset['opinion'] = ''
# Cambio stars por 0 y 1
i = 0
while i < len(dataset):
    if dataset.stars[i] < 3:
       dataset.opinion[i] = 0
    else:
       dataset.opinion[i] = 1
    i = i +1

list_reviews_lemm = list(dataset['review_lemm'].values)
list_reviews_stem = list(dataset['review_stem'].values)
stars = dataset['stars'].values
opinion = dataset['opinion'].values

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(list_reviews_lemm)

x = X.toarray()
y = opinion
y = y.astype('int')

# Armo modelo ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(x, y)

pickle.dump(etc, open("tree.pkl", "wb"))
pickle.dump(vectorizer, open("vec.pkl", "wb"))

# Armo modelo RandomForest
rf = RandomForestClassifier()
rf.fit(x, y)

pickle.dump(rf, open("rf.pkl", "wb"))


# Armo modelo Red Neuronal Keras'
lr = 0.01        
nn = [5000, 4, 1]  

model = kr.Sequential()

l1 = model.add(kr.layers.Dense(nn[0], activation='relu'))
l2 = model.add(kr.layers.Dense(nn[1], activation='relu'))
l3 = model.add(kr.layers.Dense(nn[2], activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamos al modelo
#model.fit(x, y, epochs=5, batch_size=200)
#model.save('model.h5')