import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# users_df = pd.read_csv('users.csv')
reviews_df = pd.read_csv('datasets/reviews.csv')
# details_df = pd.read_csv('details.csv')

model_directory = "model" # finetuned text-classification model dir...
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForSequenceClassification.from_pretrained(model_directory)

sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label']

reviews_df['sentiment'] = reviews_df['text'].apply(get_sentiment)

reviews_df.to_csv('datasets/reviews_with_sentiment.csv', index=False)

print("Sentiment analysis completed and saved to 'reviews_with_sentiment.csv'")

