
import numpy as np

from sklearn.neighbors import NearestNeighbors
from animals import X, Y, animal_data

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
nbrs.fit(X)

# legs, size, fur (1=yes), flying (1=yes)
guess = np.array([[4, 3, 1, 0]])

distances, indices = nbrs.kneighbors(guess)

for ix, dist in zip(indices[0], distances[0]):
    name = animal_data['name'].iloc[ix]
    print("found {:>15s} with distance {:6.2f}.".format(name, dist))

