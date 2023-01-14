import streamlit as st
from utils import check_password, load_model, get_results

st.set_page_config("Sentiment Analysis App", None, "wide")
st.write("# Sentiment Analysis App")

if check_password():
    # More info about the model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
    task = "sentiment"
    model_name = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer, model, labels = load_model(task, model_name)

    with st.form("text-input"):
        text = st.text_input("Enter text to classify")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        get_results(text, tokenizer, model, labels)
