import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from animals import X, Y

clf = RandomForestClassifier(n_estimators=10, max_depth=10)
clf = clf.fit(X, Y)

# legs, size, fur (1=yes), flying (1=yes)
cow = np.array([4, 4, 1, 0])
print(clf.predict([cow]))

scores = cross_val_score(clf, X, Y)
print(scores)