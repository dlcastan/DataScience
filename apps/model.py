import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
import keras as keras
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras


with open('vec.pkl', 'rb') as v:
    vectorizer = pickle.load(v)

def app():

   st.subheader('Selecciona las opciones:') 

   modelo = st.radio(
   "Selecciona el modelo de Machine Learning:",
   ('RandomForest', 'ExtraTreesClassifier', 'Red Neuronal Keras'))

   review = st.text_input("Ingrese el texto de su opinión", value='', max_chars=None) 

   st.subheader('')  
   boton = st.button('Ver Predicción')
   st.subheader('') 

   prediccion= ""

   def tree():
      tree = pickle.load(open("tree.pkl", 'rb'))
      prediccion = tree.predict(vectorizer.transform([review]).toarray())

      if prediccion == 1:
         prediccion = st.success("La review predicha es positiva.")
      else:
         prediccion = st.error("La review predicha es negativa.")

      return prediccion

   def rf():
      rf = pickle.load(open("tree.pkl", 'rb'))
      prediccion = rf.predict(vectorizer.transform([review]).toarray())

      if prediccion == 1:
         prediccion = st.success("La review predicha es positiva.")
      else:
         prediccion = st.error("La review predicha es negativa.")

      return prediccion
   
   def rn():
      model_kr = keras.models.load_model("model.h5")
      prediccion = model_kr.predict(vectorizer.transform([review]).toarray())

      if prediccion > 0.50 :
         prediccion = st.success("La review predicha es positiva.")
      else:
         prediccion = st.error("La review predicha es negativa.")
         
      return prediccion


   if boton:
      if review == "":
         st.text("Por favor ingrese un texto")
 
      if modelo == 'ExtraTreesClassifier':
         prediccion = tree() 
      elif modelo == 'Red Neuronal Keras':
         prediccion = rn()
      elif modelo == 'RandomForest':
         prediccion = rf()
      
      prediccion