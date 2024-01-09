from recommenders import movies, recommend_nmf

def test_movies_are_strings():
    for movie in movies:
        assert isinstance(movie,str)

