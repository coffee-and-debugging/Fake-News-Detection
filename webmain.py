import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

true_data = pd.read_csv("True.csv")
fake_data = pd.read_csv("Fake.csv")
true_data["class"] = 0
fake_data["class"] = 1
merge_data = pd.concat([true_data, fake_data], axis=0)
def word(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+www\.\S+", '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

merge_data['text'] = merge_data['text'].apply(word)
x = merge_data['text']
y = merge_data['class']

vectorization = TfidfVectorizer()
xv = vectorization.fit_transform(x)
LR = LogisticRegression()
LR.fit(xv, y)


def main():
    st.title("Fake News Detection")
    input_text = st.text_area("Enter your news article here: ")
    if st.button("Submit"):
        if input_text:
            prediction = manual_testing(input_text)
            st.write("Prediction:", prediction)

def manual_testing(news):
    cleaned_news = word(news)
    input_data = vectorization.transform([cleaned_news])
    pred = LR.predict(input_data)
    return output_label(pred[0])

def output_label(counter):
    if counter == 0:
        return "Fake News"
    elif counter == 1:
        return "Real News"

if __name__ == "__main__":
    main()