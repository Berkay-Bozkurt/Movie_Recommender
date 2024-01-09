import pandas as pd
from scipy.sparse import csr_matrix

def prepare_data(movies_path="./data/movies.csv", ratings_path="./data/ratings.csv", links_path="./data/links.csv"):
    """
    Perform data preparation by reading movie and ratings data, merging them, calculating average ratings,
    filtering for popular movies, and returning the prepared DataFrame.
    """
    try:
        df_mov = pd.read_csv(movies_path)
        df_user = pd.read_csv(ratings_path)
        df_mov_link = pd.read_csv(links_path)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return None

    df = df_mov.merge(df_user, on=["movieId"])
    df = df.merge(df_mov_link, on=["movieId"])

    df["av_rating"] = df.groupby("title")["rating"].transform("mean")

    rating_count = df.groupby('movieId')[['rating']].count()
    popular_movies = rating_count[rating_count['rating'] > 20].index
    df = df[df['movieId'].isin(popular_movies)].copy()

    return df

def transform_data(df):
    """
    Perform data transformations by mapping user and movie IDs, sorting the DataFrame,
    and extracting the list of movie titles.
    """
    user_ids = df['userId'].unique()
    user_id_map = {v: k for k, v in enumerate(user_ids)}
    df['userId'] = df['userId'].map(user_id_map)

    movie_ids = df['movieId'].unique()
    movie_id_map = {v: k for k, v in enumerate(movie_ids)}
    df['movieId'] = df['movieId'].map(movie_id_map)

    df.sort_values(by="av_rating", ascending=False, inplace=True)

    df_grp = df.groupby("title")[["av_rating"]].first()
    df_grp_links = df.groupby("title")[["tmdbId"]].first()
    movies = df_grp.index.tolist()
    links = df_grp_links.index.tolist()

    return df, movies, links

def create_csr_matrix(df):
    """
    Create a CSR matrix from the DataFrame containing user ratings.
    """
    R = csr_matrix((df['rating'], (df['userId'], df['movieId'])))
    Rt = R.todense()
    return R, Rt

# Perform data preparation and transformations, and create the CSR matrix
prepared_data = prepare_data()
if prepared_data is not None:
    df, movies, links = transform_data(prepared_data)
    R, Rt = create_csr_matrix(df)