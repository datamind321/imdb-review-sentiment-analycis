import streamlit as st 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
import pickle 
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


ps = PorterStemmer()

model = pickle.load(open('model.pkl','rb')) 
vector = pickle.load(open('vectorized.pkl','rb'))  

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)   

    
st.title('IMDB Reviews Sentiment Analycis')

review = st.text_area("Please Enter your Reviews") 

if st.button('Sentiment'):
    transform_review = transform_text(review)
    preprocess_review = vector.transform([transform_review]).toarray()
    output= model.predict(preprocess_review)
    if output == 0:
        st.image('Screenshot 2023-07-20 160555.png')
        st.error("Negative Review")
        speak("Negative Review")
    elif output ==1:
        st.image('Screenshot 2023-07-20 160545.png')
        st.success("Positive Review")
        speak("Positive Review")






