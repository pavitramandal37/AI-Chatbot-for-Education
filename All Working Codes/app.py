# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 20:21:53 2022

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

from flask import Flask, render_template, request

import random


df=pd.read_csv('data.csv')
question_token = df['questions'].values.tolist()
answer_token  = df['answers'].values.tolist()



GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
        
def preprocess_text(text):
  spell = Speller(lang= 'en')
  stop_words = stopwords.words('english')

  text=text.lower()
  re.sub(r'([^\s\w]|_)+', ' ', text)
  text=spell(text)
  text=word_tokenize(text)
  text=' '.join([j for j in text if j not in stop_words])
  return text


def response(user_response,question_token , answer_token):
    robo_response=''
    TfidfVec = TfidfVectorizer( stop_words='english')
    tfidf = TfidfVec.fit_transform(question_token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't know the answer."
        return robo_response
    else:
        robo_response = robo_response+answer_token[idx]
        return robo_response


def chatbot_response(msg):
    user_response=msg.lower()
    
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            
            res= "You are welcome.."
        else:
            if(greeting(user_response)!=None):
                
                res= greeting(user_response)
            else:
                user_response=preprocess_text(user_response)
                print("User is:",user_response)
                question_token.append(user_response)
                res=(response(user_response,question_token,answer_token ))
                question_token.pop()
                print("Last is : ",question_token[-1])
                
    else:
        res="Bye! take care.."
        
    print(res)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()















