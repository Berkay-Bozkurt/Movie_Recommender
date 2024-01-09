import pickle
import json
from recommenders import recommend_nmf, recommend_neighborhood, score_transformation

def load_models(models_list):
    """
    Load the trained models from pickle files.
    """
    with open(f'models/{models_list[0]}.pkl', 'rb') as file:
        model_1 = pickle.load(file)
    
    with open(f'models/{models_list[1]}.pkl', 'rb') as file:
        model_2 = pickle.load(file)
    
    return model_1, model_2


def get_top_movies(model, k=5):
    """
    Get the top-k movies from the recommendation model.
    
    Args:
        model (pd.Series): The recommendation scores for movies.
        k (int): The number of top movies to retrieve.
    
    Returns:
        list: The top-k movie titles.
    """
    model_list = model.sort_values(ascending=False).index.tolist()
    top_movies = model_list[:k]
    return top_movies


def combination_of_models(model1, model2):
    """
    Combine the recommendation scores from two models.
    
    Args:
        model1 (pd.Series): The recommendation scores from model 1.
        model2 (pd.Series): The recommendation scores from model 2.
    
    Returns:
        list: The combined top-k movie titles.
    """
    total_recommendation = (model1.add(model2)) / 2
    total_recommendation.dropna(inplace=True)
    total_recommendation_ = total_recommendation.sort_values(ascending=False).index.tolist()
    return total_recommendation_

user_query = json.load(open("streamlit/user_query.json"))

existing_models = ["nmf_model", "near_model"]
def main():
    nmf_model, near_model = load_models(existing_models)
    recommend_neighborhood_ = score_transformation(recommend_neighborhood(user_query, near_model, 20), 30)
    recommend_neighborhood_top = get_top_movies(recommend_neighborhood_)
    
    recommend_nmf_ = score_transformation(recommend_nmf(user_query, nmf_model), 30)
    recommend_nmf_top = get_top_movies(recommend_nmf_)
    
    combined_model = combination_of_models(recommend_neighborhood_, recommend_nmf_)

    final_recommendations = list(set(recommend_neighborhood_top + recommend_nmf_top + combined_model))
    print(final_recommendations)
    return recommend_nmf_top, recommend_neighborhood_top, combined_model


if __name__ == "__main__":
    main()