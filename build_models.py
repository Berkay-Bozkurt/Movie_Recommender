import numpy as np
import pickle
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from data_transformation import R, Rt

def build_model_nmf(n_components=2000, max_iter=2000, tol=0.0001, verbose=1):
    """
    Build and save an NMF model.
    """
    model = NMF(n_components=n_components, max_iter=max_iter, tol=tol, verbose=verbose)
    model.fit(R)

    with open('models/nmf_model.pkl', mode='wb') as file:
        pickle.dump(model, file)
    
    return 'nmf_model.pkl'

def build_model_neighbors(metric='cosine', n_jobs=-1):
    """
    Build and save a Nearest Neighbors model.
    """
    model = NearestNeighbors(metric=metric, n_jobs=n_jobs)
    model.fit(np.asarray(Rt))

    with open('models/near_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return 'near_model.pkl'

def main() -> None:
    """
    Main function to build and save models.
    """
    file_name_nmf = build_model_nmf()
    print(f"NMF model saved as {file_name_nmf}.")

    file_name_neighbors = build_model_neighbors()
    print(f"Neighborhood model saved as {file_name_neighbors}.")

if __name__ == "__main__":
    main()