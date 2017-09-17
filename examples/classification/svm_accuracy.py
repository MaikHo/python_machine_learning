
from sklearn import svm, datasets, cross_validation

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = \
     cross_validation.train_test_split(X, Y, test_size=0.4, random_state=True)

svc = svm.SVC(kernel='linear', C=1.0, probability=True).fit(X_train, Y_train)

print(svc.score(X_test, Y_test))

accuracy = cross_validation.cross_val_score(svc, X, Y, cv=5, scoring='accuracy')
print(accuracy)

# work for binary classifier only
#print(cross_validation.cross_val_score(svc, X, Y, scoring='precision'))
#print(cross_validation.cross_val_score(svc, X, Y, scoring='recall'))
