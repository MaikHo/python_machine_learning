import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from animals import X, Y

X_train, X_test, Y_train, Y_test = \
     cross_validation.train_test_split(X, Y, test_size=0.4, random_state=True)

print(X.shape, X_train.shape)

# calculate the score for a single test set
clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
print(clf.score(X_test, Y_test))

# do a five-fold cross-validation
print(cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='accuracy'))
print(cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='precision'))
print(cross_validation.cross_val_score(clf, X, Y, cv=5, scoring='recall'))

