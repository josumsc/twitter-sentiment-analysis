# Twitter Sentiment Analysis with Streamlit & Docker

![Main App View](/docs/img/main-app-view.png)

ML web app that can be deployed through Docker to analyze the sentiment of tweets.
It uses Docker for the deployment and Streamlit for the web app.

The model used is a pre-trained model from [Hugging Face](https://huggingface.co/). Concretely, it is a roBERTa model finetuned for sentiment analysis with the TweetEval benchmark. The model is available [here](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).

## Configuration

This app expects 2 untracked files to be present in the project:

- `.streamlit/secrets.toml` with the following content:

```toml
password = "your_password"
```

This file is used to protect the app with a password. You can change the password to whatever you want or even deactivate the `if check_password` in `app.py` to simply avoid this step. I prefer to have a password to avoid random people accessing the app in case I'd choose to test it on the open.

- `.env` with the following content:

```bash
API_KEY=YOUR_API_KEY
API_KEY_SECRET=YOUR_API_KEY_SECRET
BEARER_TOKEN=YOUR_BEARER_TOKEN
ACCESS_TOKEN=YOUR_ACCESS_TOKEN
ACCESS_TOKEN_SECRET=YOUR_ACCESS_TOKEN_SECRET
```

This file is used to authenticate the app to the Twitter API. You can get these credentials by creating a Twitter developer account and creating a new app. You can find more information [here](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api).

In case you just want to run the app for analyzing single tweets instead of user accounts, I have created a second prompt that does not require the Twitter API credentials as it depends solely on the model and the text you input.

![Single Tweet View](/docs/img/text-input.png)

## How to run

A `docker-compose.yml` file is provided to run the app. You can run it with:

```bash
docker-compose up
```

This should have created the webapp for you to explore and access it at `http://localhost:8501`.

Changes should be reflected in the web app automatically thanks to the volumes of the `docker-compose.yml` file.

To stop the app, you can use `Ctrl+C` or `docker-compose down` in case you are running it on daemon mode.

## References

- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)
- [TweetEval](https://arxiv.org/pdf/2010.12421.pdf)
- [Tweepy](https://docs.tweepy.org/en/stable/api.html)
- [Twitter API](https://developer.twitter.com/en/docs/twitter-api)
