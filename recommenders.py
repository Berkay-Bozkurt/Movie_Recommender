"""
In this script we define functions for the recommender web
application
"""
import pandas as pd
import numpy as np
from data_transformation import df, movies, R, Rt


def recommend_nmf(query, model):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    
    Args:
        query (dict): A dictionary containing movie ratings for the query user.
        model: The trained NMF model.
    
    Returns:
        list: The top k movie ids recommended for the query user.
    """
    
    # 1. Construct new_user-item dataframe given the query
    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix, columns=movies, index=model.get_feature_names_out())
    new_user_dataframe = pd.DataFrame(query, columns=movies, index=["new_user"])
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)

    # 2. Scoring
    # Calculate the score with the NMF model
    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    P_new_user = pd.DataFrame(P_new_user_matrix, columns=model.get_feature_names_out(), index=['new_user'])
    R_hat_new_user_matrix = np.dot(P_new_user, Q)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix, columns=movies, index=['new_user'])    

    # 3. Ranking
    # Filter out movies already seen by the user
    R_hat_new_user_filtered = R_hat_new_user.drop(query.keys(), axis=1)
    nmf_score = R_hat_new_user_filtered.sum()
    return nmf_score


def recommend_neighborhood(query, model, ratings):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    
    Args:
        query (dict): A dictionary containing movie ratings for the query user.
        model: The trained nearest neighbors model.
        ratings (int): The number of neighbors to consider for recommendation.
    
    Returns:
        list: The top k movie ids recommended for the query user.
    """
    # 1. Candidate generation
    # Construct a user vector
    new_user_dataframe = pd.DataFrame(query, columns=movies, index=['new_user'])
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)
   
    # 2. Scoring
    # Find n neighbors
    similarity_scores, neighbor_ids = model.kneighbors(new_user_dataframe_imputed,
                                                       n_neighbors=ratings,
                                                       return_distance=True)
    df_neighbors = pd.DataFrame(
        data={
            "neighbor_id": neighbor_ids[0],
            "similarity_score": similarity_scores[0],
        }
    )
    df_neighbors.sort_values("similarity_score", ascending=False, inplace=True)

    neighborhood = pd.DataFrame(Rt, columns=movies).iloc[neighbor_ids[0]]
    
    # Calculate their average rating
    neighborhood_filtered = neighborhood.drop(query.keys(), axis=1)
    df_score = neighborhood_filtered.apply(
        lambda x: df_neighbors.set_index("neighbor_id").loc[x.index][
            "similarity_score"
        ]
        * x
    )
    df_score_ranked = df_score.sum(axis=0).sort_values(ascending=False)
    return df_score_ranked


def score_transformation(score, k):
    """
    Transforms the recommendation scores and returns the top-k highest rated movie ids or titles.
    
    Args:
        score (pd.Series): The recommendation scores for movies.
        k (int): The number of top recommendations to return.
    
    Returns:
        pd.Series: The top-k highest rated movie ids or titles.
    """
    recommendations = score.sort_values(ascending=False)
    amin, amax = min(recommendations), max(recommendations)
    for i, val in enumerate(recommendations):
        recommendations[i] = (val - amin) / (amax - amin)
    # Return the top-k highest rated movie ids or titles
    return recommendations[:k]