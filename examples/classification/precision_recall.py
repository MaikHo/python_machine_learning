
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score
from animals import X, Y, animal_data

gnb = GaussianNB()
gnb.fit(X, Y)

Ypred = gnb.predict(X)

# calculate precision + recall

Ypred = gnb.predict(X)

print("accuracy : {:6.2f}".format(accuracy_score(Y, Ypred)))
print("precision: {:6.2f}".format(precision_score(Y, Ypred)))
print("recall   : {:6.2f}".format(recall_score(Y, Ypred)))
