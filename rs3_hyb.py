import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from textblob import TextBlob

users = pd.read_csv('datasets/users.csv')
reviews = pd.read_csv('datasets/reviews.csv')
details = pd.read_csv('datasets/details.csv')

details['description'] = details['description'].fillna('')
details['web_url'] = details['web_url'].fillna('')
details['city'] = details['city'].fillna('')
details['state'] = details['state'].fillna('')
details['country'] = details['country'].fillna('')
details['amenities'] = details['amenities'].fillna('')
details['styles'] = details['styles'].fillna('')
details['ranking'] = details['ranking'].fillna(0)
details['rating'] = details['rating'].fillna(0)
details['num_reviews'] = details['num_reviews'].fillna(0)
details['price_level'] = details['price_level'].fillna('')
reviews['title'] = reviews['title'].fillna('')
reviews['text'] = reviews['text'].fillna('')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(reviews[['user_id', 'location_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

algo = SVD()

algo.fit(trainset)

tfidf = TfidfVectorizer(stop_words='english')
reviews['combined_text'] = reviews['title'] + ' ' + reviews['text']
tfidf_matrix = tfidf.fit_transform(reviews['combined_text'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_user_reviews(user_id):
    return reviews[reviews['user_id'] == user_id].index.tolist()

def get_similar_reviews(user_id):
    user_reviews_idx = get_user_reviews(user_id)
    sim_scores = []

    for idx in user_reviews_idx:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:10]
    similar_reviews_idx = [i[0] for i in sim_scores]
    return reviews.iloc[similar_reviews_idx]

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

reviews['sentiment'] = reviews['text'].apply(get_sentiment)

def hybrid_recommendations(user_id):
    user_ratings = reviews[reviews['user_id'] == user_id]
    user_ratings = user_ratings[['location_id', 'rating']].groupby('location_id').mean().reset_index()
    user_ratings = user_ratings.sort_values('rating', ascending=False)
    
    similar_reviews = get_similar_reviews(user_id)
    similar_locations = similar_reviews['location_id'].value_counts().reset_index()
    similar_locations.columns = ['location_id', 'count']
    similar_locations = similar_locations.sort_values('count', ascending=False)
    
    hybrid_recs = pd.merge(user_ratings, similar_locations, on='location_id', how='outer').fillna(0)
    hybrid_recs = pd.merge(hybrid_recs, reviews[['location_id', 'sentiment']], on='location_id', how='left').fillna(0)
    
    hybrid_recs['score'] = hybrid_recs['rating'] + hybrid_recs['count'] + hybrid_recs['sentiment']
    hybrid_recs = hybrid_recs.sort_values('score', ascending=False)
    
    return hybrid_recs[['location_id', 'score']]


user_id = 986
recommendations = hybrid_recommendations(user_id)
print(recommendations)
