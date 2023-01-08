import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from utils import check_password, get_labels, predict

st.set_page_config("Sentiment Analysis App", None, "wide")
if check_password():
    st.title("Welcome to the app!")
    with st.spinner("Loading Model Into Memory..."):
        # More info about the model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        labels = get_labels(task)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    text = st.text_input("Enter text to classify")
    scores = predict(text, tokenizer, model)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    st.write("Results for your tweet:")

    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
