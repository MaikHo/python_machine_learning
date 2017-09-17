
import numpy as np
from sklearn.naive_bayes import GaussianNB
from animals import X, Y, animal_data

gnb = GaussianNB()
gnb.fit(X, Y)

# legs, size, fur (1=yes), flying (1=yes)
dolphin = np.array([0, 4, 0, 0])
print(gnb.predict([dolphin]))


Ypred = gnb.predict(X)
animal_data['predicted'] = Ypred
print(animal_data[animal_data['reproduction']!=Ypred])

print("Number of mislabeled points out of {} points : {}".format(
    X.shape[0], (Y != Ypred).sum()))

