import streamlit as st
from scipy.special import softmax
import csv
import urllib.request
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


def get_labels(task):
    """Download the labels for a given task from the TweetEval repository."""
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    return labels


def preprocess(text):
    """Preprocess the text to be classified."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def predict(text, tokenizer, model):
    """Predict the score of the labels for the given text."""
    preprocessed_text = preprocess(text)
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt')

    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores


@st.experimental_singleton(show_spinner=False)
def load_model(task, model_name):
    """Retrieves the model from Hugging Face's model hub and loads it into memory."""
    with st.spinner("Loading Model Into Memory..."):
        labels = get_labels(task)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model, labels


def get_results(text, tokenizer, model, labels):
    """Writes in our webapp the results obtained from the model."""
    scores = predict(text, tokenizer, model)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    st.write("Results for your tweet:")

    result_df = pd.DataFrame(zip(labels, scores), columns=['Label', 'Score'])
    st.table(result_df.sort_values(by='Score', ascending=False).reset_index(drop=True))
