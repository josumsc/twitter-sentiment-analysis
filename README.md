# Twitter Sentiment Analysis with Streamlit & Docker

ML web app that can be deployed through Docker to analyze the sentiment of tweets.
It uses Docker for the deployment and Streamlit for the web app.

The model used is a pre-trained model from [Hugging Face](https://huggingface.co/). Concretely, it is a roBERTa model finetuned for sentiment analysis with the TweetEval benchmark. The model is available [here](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment).

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
