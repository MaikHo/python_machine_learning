
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from animals import X, Y, FEATURE_NAMES, CLASS_NAMES
import numpy as np


def export_tree(clf, filename):
    # export tree file
    from sklearn.externals.six import StringIO
    with open("{}.dot".format(filename), 'w') as f:
         f = tree.export_graphviz(clf, out_file=f,
                            feature_names=FEATURE_NAMES,  
                            class_names=CLASS_NAMES, 
                            )

    # wait for the file to be written
    import time
    time.sleep(1)

    # simplify the graph
    import re
    text = open('{}.dot'.format(filename)).read()
    text = re.sub('gini\s\=\s\d+\.\d+', '', text)
    text = re.sub('value\s\=\s\[\d+, \d+\]', '', text)
    open('{}.dot'.format(filename), 'w').write(text)

    # requires installation of Graphviz
    import os
    os.system('dot -Tpdf {0}.dot -o {0}.pdf'.format(filename))


if __name__ == '__main__':
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y)
    print(scores)

    # legs, size, fur (1=yes), flying (1=yes)
    cow = np.array([4, 4, 1, 0])
    print(clf.predict([cow]))

    export_tree(clf, 'animal_tree')
