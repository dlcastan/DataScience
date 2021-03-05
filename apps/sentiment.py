import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from textblob import TextBlob

def app():
    st.subheader('Selecciona las opciones:') 
    review = st.text_input("Ingrese el texto de su opinión", value='', max_chars=None) 
    
    st.subheader('')  
    boton = st.button('Procesar')

    if boton:
      if review == "":
        st.text("Por favor ingrese un texto")
      else:  
        t=TextBlob(review)
        ten=t.translate(to="en")
        ten.sentiment
        st.text("")
   
        st.subheader('Resultado:') 

        if ten.sentiment[0] > 0:
            st.success('La review es Positiva')
        else:
            st.error('La review es Negativa')

        if ten.sentiment[1] > 0:
            st.warning('La review es una Opinion')
        else:
            st.success('La review es un Hecho')