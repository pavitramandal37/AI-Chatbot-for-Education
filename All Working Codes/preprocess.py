# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 20:25:56 2022

@author: upjab
"""
import json
import os

import pandas as pd
import nltk
import re
from autocorrect import Speller
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

with open('data.json') as json_file:
    data = json.load(json_file)
    
data = data['data'][0]
data=data['paragraphs']

for i in data:
  d=i['qas']

questions=[]
answers=[]

for j in d:
  questions.append(j['question'])
  for k in j['answers']:
    answers.append(k['text'])
print('Done')

df=pd.DataFrame({'questions':questions,'answers':answers})

def preprocess(df):
  spell = Speller(lang= 'en')
  stop_words = stopwords.words('english')

  for idx , row in df.iterrows():
    row['questions']=row['questions'].lower()
    row['answers']=row['answers'].lower()
    re.sub(r'([^\s\w]|_)+', ' ', row['questions'])
    re.sub(r'([^\s\w]|_)+', ' ', row['answers'])
    
    row['questions'] = spell(row['questions'])
    row['answers'] = spell(row['answers'])
    
    row['questions'] = word_tokenize(row['questions'])
      
    row['questions'] = ' '.join([j for j in row['questions'] if j not in stop_words])
    
  print('Done')
  return df
  
preprocess(df)
df.to_csv('data.csv')
