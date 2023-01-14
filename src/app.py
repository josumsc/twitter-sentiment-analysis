import streamlit as st
from utils import (
    check_password,
    load_model,
    get_tweepy_api,
    get_tweets_from_username,
    get_results_df_from_tweets,
    summarize_tweet_profiling,
    get_text_results,
)

st.set_page_config("Sentiment Analysis App", None, "wide")
st.write("# Sentiment Analysis App")

if check_password():
    # More info about the model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
    task = "sentiment"
    model_name = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer, model, labels = load_model(task, model_name)

    with st.form("username-input"):
        username = st.text_input("Enter **@username** to classify")
        submit_button_username = st.form_submit_button("Submit")

    if submit_button_username:
        api = get_tweepy_api()
        with st.spinner("Loading tweets..."):
            tweets = get_tweets_from_username(username, api)
        with st.spinner("Classifying tweets..."):
            results_df = get_results_df_from_tweets(tweets, tokenizer, model)
        summarize_tweet_profiling(username, results_df)

    else:
        st.write("Or...")
        with st.form("text-input"):
            text = st.text_input("Copy-paste tweet to classify")
            submit_button_text = st.form_submit_button("Submit")
        
        if submit_button_text:
            get_text_results(text, tokenizer, model, labels)
