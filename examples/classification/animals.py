
# predict reproduction mode of animals
# 0=eggs, 1=live birth

import pandas as pd
animal_data = pd.read_csv("animals.txt", header=0)

FEATURE_NAMES = ['reproduction','size','fur','flying']
CLASS_NAMES = ['0','1','2','3','4']#['eggs', 'live_birth']

X = animal_data[FEATURE_NAMES]
Y = animal_data['legs']

