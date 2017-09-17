import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('guerry.csv')

#LABELS = ["Crime_pers","Crime_prop","Literacy","Donations",
#    "Infants","Suicides", "Wealth","Commerce","Clergy",
#    "Crime_parents","Infanticide","Donation_clergy","Lottery",
#    "Desertion","Instruction","Prostitutes","Distance","Area",
#    "Pop1831"]

LABELS = ["Literacy", "Wealth", "Clergy", "Pop1831"]

columns = data[LABELS]
cluster = KMeans(n_clusters=5, max_iter=300, verbose=0, 
                    random_state=None)
cluster.fit(columns)

# print cluster table
print(''.join(['Cluster # ']+["{:>10s}".format(x) for x in LABELS]))

for i, center in enumerate(cluster.cluster_centers_):
    print("{:10d}".format(i+1), end="")
    for value in center:
        print("{:10.1f}".format(value), end="")
    print()
# Also see: http://scikit-learn.org/stable/modules/clustering.html