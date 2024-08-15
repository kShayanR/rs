
# - pip install pandas scikit-learn
# - pip install numpy<2
# - pip install scikit-surprise
# - pip install textblob


#################--- Collaborative ---#################

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate

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

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(reviews[['user_id', 'location_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

algo = SVD()

algo.fit(trainset)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

def predict_rating(user_id, location_id):
    prediction = algo.predict(user_id, location_id)
    return prediction.est

user_id = 912  
location_id = 4874757
predicted_rating = predict_rating(user_id, location_id)
print(f'Predicted rating for user {user_id} on hotel {location_id}: {predicted_rating}')

#################--- Content-Based ---#################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf = TfidfVectorizer(stop_words='english')
details['description'] = details['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(details['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(name, cosine_sim=cosine_sim):
    idx = details[details['name'] == name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10 
    hotel_indices = [i[0] for i in sim_scores]
    return details['name'].iloc[hotel_indices]


print(get_recommendations('Rixos Bab Al Bahr'))  
