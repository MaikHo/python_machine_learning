
# Authors: Gael Varoquaux
#          Jaques Grobler
#          Kevin Hughes
# License: BSD 3 clause

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

data = pd.read_csv('guerry.csv')

columns = data[["Crime_pers","Crime_prop","Literacy","Donations",
    "Infants","Suicides", "Wealth","Commerce","Clergy",
    "Crime_parents","Infanticide","Donation_clergy","Lottery",
    "Desertion","Instruction","Prostitutes","Distance","Area",
    "Pop1831"]]

#!! Average needs to be zero for PCA to work
mean = columns.mean()
demeaned = columns - mean

pca = PCA(n_components=1)
pca.fit(demeaned)

print(pca.components_)
print(pca.explained_variance_ratio_)
