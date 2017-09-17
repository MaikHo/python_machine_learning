
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from animals import X, Y, animal_data

nbrs = KNeighborsClassifier(n_neighbors=3, weights='distance')
nbrs.fit(X, Y)

# legs, size, fur (1=yes), flying (1=yes)
ostrich = np.array([[2, 4, 0, 1]])
print(nbrs.predict(ostrich))

mouse = np.array([[4, 2, 1, 0]])
print(nbrs.predict(mouse))

