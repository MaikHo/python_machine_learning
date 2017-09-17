
# Non-negative Matrix Factorization (NMF)

# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
# http://vazic.me/non-negative-matrix-factorization-nmf/

# https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf

import numpy as np
from sklearn.decomposition import NMF
import pandas as pd

# read matrix R
ratings = pd.read_csv('movies.csv', index_col=0)
X = ratings.values

# R : user rating
# R ~ WH'

model = NMF(n_components=2, init='random', random_state=10)
model.fit(X)

# H : movie feature
H = model.components_

# W : user feature
W = model.transform(X)
print(ratings)
print(W)

print(model.reconstruction_err_)

nR = np.dot(W,H)
print(nR)

query = [0,0,5,0]
print(model.transform(query))
