import streamlit as st
from multiapp import MultiApp
from PIL import Image
from apps import model, process, sentiment
from PIL import Image

app = MultiApp()

image = Image.open('acamica_logo.jpg')
st.image(image, use_column_width=True)

st.title('Análisis de los textos')

app.add_app("Procesamiento", process.app)
app.add_app("Sentimiento", sentiment.app)
app.add_app("Modelo", model.app)

app.run()