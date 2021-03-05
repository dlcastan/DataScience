import streamlit as st
import numpy as np
import pandas as pd
import spacy
from spacy.lang.es import Spanish
import es_core_news_sm


def app():
    st.title('Procesamiento')

    st.subheader('Selecciona las opciones:')
    
    opcion = st.radio(
   "Selecciona una opción",
   ('Tokenizar', 'Lematizar', 'Tipo Palabras','Entidades'))

    review = st.text_input("Ingrese el texto de su opinión", value='', max_chars=None) 

    st.subheader('')  
    boton = st.button('Procesar')

    def lem(my_text):
        nlp = es_core_news_sm.load()
        docx = nlp(my_text)
        tokens = [token.text for token in docx]
        allData = [(token.text, token.lemma_) for token in docx]
        return allData

    def entity(my_text):
        nlp = es_core_news_sm.load()
        docx = nlp(my_text)
        tokens = [ token.text for token in docx]
        entities = [(entity.text,entity.label_) for entity in docx.ents]
        allData = [(entities)]
        return allData

    def token(my_text):
        nlp = es_core_news_sm.load()
        docx = nlp(my_text)
        tokens = [ token.text for token in docx]
        allData = [tokens]
        return allData

    def tipo(texto):
        nlp = es_core_news_sm.load()
        text = nlp(texto)
        tokens = [token.text for token in text]
        doc = [(token.text, token.tag_) for token in text]
        return doc

    if boton:
        if review == "":
            st.text("Por favor ingrese un texto")
        else:  
            if opcion=="Tokenizar":
                st.write('')
                result = token(review)
                st.success('Textos Tokenizados')
                st.success(result)
            elif opcion=="Lematizar":
                st.write('')
                result = lem(review)
                st.success('Textos Lematizados')
                st.success(result)
            elif opcion=="Tipo Palabras":
                st.write('')
                st.success('Tipo Palabras')
                result = tipo(review)
                st.success(result)
            elif opcion=="Entidades":
                st.write('')
                result = entity(review)
                st.success('Entidades')
                st.success(result)
            else:
                st.text("Error opción")