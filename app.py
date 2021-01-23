import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from autocorrect import Speller
from google_trans_new import google_translator  

from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import dump, load 
import gdown
import socket

sns.set_style('darkgrid')
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='COVID-Fact-checker', page_icon=None, initial_sidebar_state='auto')

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
lemmatizer = WordNetLemmatizer() 
stopwords_list = pd.read_csv('Datos/my_stopwords.txt', names=['my_stopwords'])
stopwords_list = stopwords_list['my_stopwords'].values.tolist()
spell = Speller(lang='en', fast=True)

def prepare_text(text, predict=True):
    # Convert to lowercase
    clean_text = text.lower()
    # Remove foreign characters and emojis
    clean_text = re.sub('[^\u0020-\u024F]', '', clean_text)
    # Remove urls
    clean_text = re.sub(r'http\S+', '', clean_text)
    # Remove punctuation, keep @ because TweetTokenizer handles it
    clean_text = re.sub('[!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~]', '', clean_text)
    # Remove whitespaces
    clean_text = re.sub('[\t\n\r\x0b\x0c]', '', clean_text)
    # Spell check
    clean_text = spell(text)
    # Tokenize
    clean_text = tokenizer.tokenize(clean_text)
    # Remove stopwords
    clean_text = [word for word in clean_text if word not in stopwords_list]
    # Lemmatizer
    clean_text = [lemmatizer.lemmatize(word) for word in clean_text]
    # Put together again
    clean_text = ' '.join(clean_text)
    
    return clean_text

@st.cache(allow_output_mutation=True)
def load_word_vectorizer():
	# The word_vectorizer.joblib is kinda big for GitHub, so we will download it from my Drive
	url = 'https://drive.google.com/uc?id=1o8pi_BywNIBkgKRCMWRzB0KQeas6vERg&export=download'
	output = 'word_vectorizer.joblib'
	gdown.download(url, output, quiet=False)
	return load('word_vectorizer.joblib')


word_vectorizer = load_word_vectorizer()
model = load('Modelos/SVM_linear.joblib')

# Formato

st.image('Images/unam.png', use_column_width=False, width=100)
st.title('COVID Fact-Checker')

st.sidebar.image('Images/coronavirus.jpg', use_column_width=True)
st.sidebar.info('App creada por David Guzmán')

select_prediction = st.sidebar.selectbox('¿Cómo quieres checar?', ['Online', 'Batch'])

# Contenido

if select_prediction == 'Online':
	language = st.selectbox('Idioma', ['Inglés', 'Español'])
	text = st.text_input(label='Texto', value='', max_chars=150)

	if language == 'Español':
		translator = google_translator()
		prepared_text = translator.translate(text, lang_src='es', lang_tgt='en')
		st.text('Translation: {}'.format(prepared_text))
		prepared_text = prepare_text(prepared_text)
		prepared_text = word_vectorizer.transform([prepared_text])
	else:
		prepared_text = prepare_text(text)
		prepared_text = word_vectorizer.transform([prepared_text])

	prediction = model.predict(prepared_text)[0]

	if text:
		if prediction == 1:
			st.success('Parece ser cierto')
		else:
			st.error('Parece ser falso')

			"""
			Encuentra más información en: 
			- [OMS: orientaciones para el público](https://www.who.int/en/emergencies/diseases/novel-coronavirus-2019/advice-for-public)
			"""

		
else:
	language = st.selectbox('Idioma', ['Inglés', 'Español'])
	file_upload = st.file_uploader('Sube el archivo csv', type=['csv'])

	def highlight_max(s):
	    '''
	    highlight fake and real in the dataframe.
	    '''
	    is_fake = (s == 'fake')
	    return ['background-color: #FA0505' if v else 'background-color: #32D119' for v in is_fake]

	if file_upload:
		data = pd.read_csv(file_upload, names=['text'])
		if language == 'Español':
			translator = google_translator()
			data['text_es'] = data['text']
			data['text'] = data['text_es'].apply(lambda x: translator.translate(x, lang_src='es', lang_tgt='en'))


		# Si se cambia el idioma se debe cambiar el archivo
		try:
			prepared_text = data['text'].apply(lambda x: prepare_text(x)).tolist()
			prepared_text = word_vectorizer.transform(prepared_text)
			predictions = model.predict(prepared_text)
			data['label'] = predictions
			data['label'] = data['label'].replace({0: 'fake', 1: 'real'})

			if language == 'Español':
				st.write(data[['text_es', 'label']].style.apply(highlight_max, subset=['label']))
			else:
				st.write(data[['text', 'label']].style.apply(highlight_max, subset=['label']))

			if 'fake' in data['label'].tolist():
				"""
				Encuentra más información en: 
				- [OMS: orientaciones para el público](https://www.who.int/en/emergencies/diseases/novel-coronavirus-2019/advice-for-public)
				"""

		except:
			st.error('Sube de nuevo el archivo')


"""
## Necesidad de fact-checkers
"""

text = """
<div style="text-align: justify"> 
Hoy en día, las redes sociales proporcionan un terreno particularmente fértil para la difusión de información errónea: carecen de controles y regulaciones, 
los usuarios publican contenido sin tener que pasar por un editor o de revisión por pares, verificación de calificación o suministro de fuentes, y las 
redes sociales tienden a crear <em>cámaras de eco</em> o redes cerradas de comunicación aisladas de los desacuerdos.
<br><br>
El impacto de la información errónea que rodea a la pandemia de COVID-19 puede ser especialmente dañino, ya que cualquier paso en falso puede aumentar las 
posibilidades de una propagación exponencial de la enfermedad o muerte accidental debido a la automedicación, entre otros problemas.
</div>
<br><br><br>
"""

st.markdown(text, unsafe_allow_html=True)

# ¿Pongo algo relacionado a https://www.vox.com/recode/2020/5/11/21254889/twitter-coronavirus-covid-misinformation-warnings-labels?
st.image('Images/tweet_1.jpeg', use_column_width=True, caption='Ejemplo de un tweet con información falsa en el que se ha puesto una advertencia acerca de posible información errónea.')
st.markdown('<br><br><br>', unsafe_allow_html=True)
st.image('Images/tweet_2.png', use_column_width=True, caption='Otro tweet con información falsa.')

text = """
<div style="text-align: justify">
Para abordar el aumento en tiempo real de información errónea en las redes sociales relacionada con la pandemia de COVID-19 los métodos tradicionales para 
anotar cada afirmación no son escalables con la magnitud de las conversaciones que rodean la pandemia. Por tanto, la pandemia de COVID-19 ha creado una necesidad 
urgente de herramientas para combatir la propagación de información errónea.
<br><br>
De acuerdo a lo anterior, el propósito del estudio es tratar de colaborar con herramientas que puedan facilitar la detección de información falsa relacionada al 
COVID-19. Particularmente, es deseable crear herramientas en otros idiomas que no sean inglés, ya que la mayor parte de los estudios se han centrado únicamente en 
detectar información falsa en inglés, dejando de lado otras lenguas. 
</div>
"""

st.markdown(text, unsafe_allow_html=True)

"""
## Datos
"""

text = """
<div style="text-align: justify">
Los datos se encuentran en  <a href="https://www.kaggle.com/arashnic/covid19-fake-news?select=NewsRealCOVID-19_tweets_5.csv" target="_blank">COVID-19 Fake News Dataset</a>, 
tiene afirmaciones cortas relacionadas con el COVID-19 que provienen de diversos sitios de noticias y Twiter que han sido clasificadas como verídicas o falsas.
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)


table = """
<style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;
  word-break:normal;}
.tg th{background-color:#f0f0f0;border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-dzk6{background-color:#f9f9f9;text-align:center;vertical-align:top}
.tg tg.center {
    margin-left: auto;
    margin-right: auto;
}
</style>
<table class="tg" style="margin-left:auto;margin-right:auto;">
<thead>
  <tr>
    <th class="tg-amwm">Dataset limpio</th>
    <th class="tg-amwm">Estadísticas</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-dzk6"> # Afirmaciones</td>
    <td class="tg-dzk6">75,914</td>
  </tr>
  <tr>
    <td class="tg-baqh">% Afirmaciones <span style="font-style:italic">verdaderas</span></td>
    <td class="tg-baqh">95.54 %</td>
  </tr>
  <tr>
    <td class="tg-dzk6">% Afirmaciones <span style="font-style:italic">falsas</span></td>
    <td class="tg-dzk6">4.46 %</td>
  </tr>
</tbody>
</table>

"""

st.markdown(table, unsafe_allow_html=True)

"""
### Exploración de los datos
"""

text = """
<div style="text-align: justify">
En la imagen se observan las palabras más frecuentes usadas en los tweets. Si es rojo aparece con mayor frecuencia en tweets falsos, si es verde en tweets verídicos 
y gris aparece frecuentemente en ambos.
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)
st.image('Images/cloud.png', use_column_width=True)

text = """
<br>
<div style="text-align: justify">
Un problema de los datos es que la mayoría vienen de Estados Unidos, lo que puede sesgar el contenido que tienen. 
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)

st.image('Images/Tweets_by_country.png', use_column_width=True)
st.markdown('<br><br>', unsafe_allow_html=True)
tweet_map = open('Images/Tweet_map.html', 'r', encoding='utf-8').read()
components.html(tweet_map, width=None, height=500)

text = """
<br>
<div style="text-align: justify">
Otro problema es que los datos más recientes son de finales de agosto de 2020, a pesar de que sólo son de hace unos meses la información acerca del COVID-19
ha cambiado de manera rápida desde el inicio de la pandemia, por lo es necesario recolectar datos más recientes si es posible.
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)

st.image('Images/tweet_time.png', use_column_width=True)

"""
## Modelos
"""

text = """
<div style="text-align: justify">
<br>
Para preparar el texto para el modelo adicionalmente removí <em>stop words</em>, signos de puntuación, apliqué un tokenizador y lematizador. Posteriormente lo pasé a su 
representación de bola de palabras y apliqué la transformación <b>TF-IDF</b>. Una vez hecho eso apliqué el algoritmo de SMOTE para que los datos no estuvieran tan 
desbalanceados y estuvieran en una proporción de verídicos (70%) y falsos (30%). 
<br><br>
Posteriormente prové con varios clasificadores, el que mejor se desempeñaba era una SVM con kernel <i>lineal</i> y <i>radial</i>, aunque el kernel <i>lineal</i> es mucho más
rápido para predecir y por lo tanto preferible para el despliegue. 
</div>
<br>
"""

st.markdown(text, unsafe_allow_html=True)

table = """
<style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;
  word-break:normal;}
.tg th{background-color:#f0f0f0;border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg .tg-a2cf{font-family:Arial, Helvetica, sans-serif !important;;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-0vih{background-color:#f9f9f9;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-dzk6{background-color:#f9f9f9;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg" style="margin-left:auto;margin-right:auto;">
<thead>
  <tr>
    <th class="tg-baqh"></th>
    <th class="tg-a2cf">precision</th>
    <th class="tg-a2cf">recall</th>
    <th class="tg-a2cf">f1-score</th>
    <th class="tg-a2cf">support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0vih">fake</td>
    <td class="tg-dzk6">0.91</td>
    <td class="tg-dzk6">0.91</td>
    <td class="tg-dzk6">0.91</td>
    <td class="tg-dzk6">677</td>
  </tr>
  <tr>
    <td class="tg-amwm">real</td>
    <td class="tg-baqh">0.99</td>
    <td class="tg-baqh">0.99</td>
    <td class="tg-baqh">0.99</td>
    <td class="tg-baqh">14506</td>
  </tr>
</tbody>
</table>
"""

st.markdown(table, unsafe_allow_html=True)
st.markdown('<br><br>', unsafe_allow_html=True)

cf = r"""
C = \left( \begin{matrix}
614 & 63 \\ 
62 & 14444
\end{matrix} \right)
"""

st.latex(cf)

"""
## Trabajo futuro
"""

text = """
<div style="text-align: justify">
El modelo funciona pero claramente podría mejorar, especialmente porque en la bolsa de palabras no hay una noción de orden. Para mejorar se podría usar una LSTM, 
como se muestra en  

<ul>
  <li><a href="https://www.tensorflow.org/tutorials/text/text_classification_rnn" target="_blank">Text classification with an RNN</a></li>
  <li><a href="https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html" target="_blank">Text Classification with TorchText</a></li>
</ul>
</div>
"""

st.markdown(text, unsafe_allow_html=True)