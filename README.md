# DataScience Interactive Multi Web App con Streamlit
Multi Web App creada con Streamlit para manejar textos. Con la aplicación vas a poder toketizar, lematizar e identificar el tipo de letra y las entidades detectadas.
También al introducir un texto vas a poder ver si la review es negativa o positiva y si se trata de una opinión o de un hecho.
Por último vas a poder elegir entre tres modelos, para determinar si la review ingresada es positiva o negativa.
Los modelos que utilizo en la App son Random Forest, Extra Trees Classifier y Red Neuronal Keras.

Aplicación creada por [Diego Lopez Castan](https://www.diegolopezcastan.com/).

## Preview de la App
![Preview de la App |635x380](data/example.gif)

## Instalación
Dependencias que tenes que tener:
```console
pip install streamlit
pip install scikit-learn
pip install tensorflow
pip install textblob
python -m spacy download es_core_news_sm
```

## Generación de los modelos
Correr 
```console
streamlit run dev.py

## Correr App
Correr
```console
streamlit run app.py