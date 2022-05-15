# Main libraries
import os
import pandas as pd
import numpy as np
# Libraries for recommendation systems
from collections import defaultdict
from surprise import SVD
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from flask import jsonify
from flask import Flask
import json 
from bson import json_util
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
from sklearn.metrics.pairwise import cosine_similarity

def popularity_rec(data):
    ratings_mean_count = pd.DataFrame(data.groupby('productname')['rate'].mean())
    ratings_mean_count['rating_counts'] = data.groupby('productname')['rate'].count()
    ratings_mean_count = ratings_mean_count.sort_values(by=['rate','rating_counts'], ascending=[False,False])
    return ratings_mean_count.to_json()

# Objective: To get top_n recommendation for each user
def get_top_n(predictions, n=5):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# fit and predict using svd
def svd_func(train, test):
    svd = SVD(random_state=612)
    svd.fit(train)
    svd_pred = svd.test(test)
    return svd_pred, svd


def knn_item(train, test):
    knn_i = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
    knn_i.fit(train)
    knn_i_pred = knn_i.test(test)
    return knn_i_pred, knn_i

# fit and predict using knn
def knn_user(train, test):
    knn_u = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
    knn_u.fit(train)
    knn_u_pred = knn_u.test(test)
    return knn_u_pred, knn_u

def get_database():
    from pymongo import MongoClient
    import pymongo

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb+srv://ToanDang:123@mobile.st5aw.mongodb.net/Gotech?retryWrites=true&w=majority"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    from pymongo import MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['Gotech']

def getdata():
    dbname = get_database()
    collection_name = dbname["FeedbackCrawl"]
    item_details = list(collection_name.find())
    return json.dumps(item_details, default=json_util.default)

def data():
    #Loading Data files
    review_1 = pd.read_csv('feedback1_notnull.csv')
    review_2 = pd.read_csv('feedback2_notnull.csv')
    #Merge the data into a single dataframe 
    reviews = pd.concat([review_1,review_2], ignore_index=True)
    del review_1, review_2

    unknowns = ['anonymous']
    reviews['username'].replace(to_replace = unknowns,value = 'Anonymous',inplace=True)

    orgnl_rows = reviews.shape[0]
    orgnl_columns = reviews.shape[1]

    revs1 = reviews.copy()

    # Delete data which is not useful anymore, to save memory
    del reviews

    relevant_features=['username','productname','rate']
    # Step1: remove irrelevant features
    revs1 = revs1.loc[:,relevant_features]

    # Step2: Round-off score feature to nearest integer
    revs1['rate'] = revs1['rate'].round(0).astype('Int64')

    # Step3: Impute missing values in score feature with median
    revs1['rate'] = revs1['rate'].fillna(revs1['rate'].median())

    # Step4: remove samples with missing values in 'Product' and 'author' feature and also 'Anonymous' values
    revs1.dropna(inplace=True)
    revs1 = revs1[revs1["username"] != 'Anonymous']

    # Step5: remove duplicates, if any
    revs1 = revs1.drop_duplicates()
    return revs1

def popularity_rec(data):
    ratings_mean_count = pd.DataFrame(data.groupby('productname')['rate'].mean())
    ratings_mean_count['rating_counts'] = data.groupby('productname')['rate'].count()
    ratings_mean_count = ratings_mean_count.sort_values(by=['rate','rating_counts'], ascending=[False,False])
    return ratings_mean_count.head()

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return x['Type'] + ' ' + x['CategoryName'] + ' '+ x['ProductName']


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(df,title):
    metadata = df.reset_index()
    count = CountVectorizer(stop_words= None)
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df.index, index=df['product'])
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['product'].iloc[movie_indices].to_json()

@app.route('/popu')
def popu():
    revs = data()
    df = popularity_rec(revs)
    return df.to_json()

@app.route('/cb')
def cb():
    df = pd.read_csv('Products2.csv')
    df['ProductId'] = df['_id']
    df['product'] = df['ProductName']
    df = df.drop('_id', 1)

    features = [ 'CategoryName', 'ProductName', "Type"]
    for feature in features:
        df[feature] = df[feature].apply(clean_data)

    df['soup'] = df.apply(create_soup, axis=1)
    res = get_recommendations(df, 'REDMI 9i Sport (Metallic Blue, 64 GB)')
    return res
@app.route('/')
def index():
    # create contants
    RS=612
    revs1 = data()

    # 3. Select data with products having >1000 ratings and users who have given > 50 ratings
    author50 = revs1['username'].value_counts()
    author50 = author50[author50>50].index.tolist() # list of authors with > 50 ratings
    product50 = revs1['productname'].value_counts()
    product50 = product50[product50>1000].index.tolist() # list of products with > 1000 ratings
    revs_50 = revs1[(revs1['username'].isin(author50)) & (revs1['productname'].isin(product50))]
    del author50, product50


    # Rearrange columns for SVD and prepare train and testsets
    revs50_ = Dataset.load_from_df(revs_50[['username','productname','rate']], Reader(rating_scale=(1, 10)))
    trainset, testset = train_test_split(revs50_, test_size=.25,random_state=RS)

    # svd_pred, svd = svd_func(trainset,testset)
    # svd_rmse = round(accuracy.rmse(svd_pred),2)


    # knn_i_pred, knn_i = knn_item(trainset, testset)
    # knn_i_rmse = round(accuracy.rmse(knn_i_pred),2)
    
    knn_u_pred, knn_u = knn_user(trainset, testset)
   
    top_5 = get_top_n(knn_u_pred,5)
    # for key,value in top_5.items():  
    #     dbname = get_database()
    #     collection_name = dbname["colap"]
    #     document = {key: value}
    #     collection_name.insert_many([document])
    return jsonify(top_5)

if __name__ == "__main__":   
    app.run(debug=True)

