import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import maxabs_scale

from sklearn.ensemble import ExtraTreesClassifier


dataSet = pd.read_csv("blood.txt")

f1n = np.array(dataSet["Recency (months)"])
f1 = f1n.reshape(-1,1)


f2n = np.array(dataSet["Frequency (times)"])
f2 = f2n.reshape(-1,1)


f3n = np.array(dataSet["Monetary (c.c. blood)"])
f3 = f3n.reshape(-1,1)


f4n = np.array(dataSet["Time (months)"])
f4 = f4n.reshape(-1,1)

label = np.array(dataSet["donate"])

features = np.concatenate((f1,f2,f3,f4), axis=1)

f_train, f_test, l_train, l_test = train_test_split(features, label, test_size=0.2)

min_max_scale_train = minmax_scale(f_train)
scaled_features_train = maxabs_scale(min_max_scale_train)

min_max_scale_test = minmax_scale(f_test)
scaled_features_test = maxabs_scale(min_max_scale_test)


blood_classifier = ExtraTreesClassifier(n_estimators=1000)

blood_classifier.fit(scaled_features_train, l_train)

result = blood_classifier.predict(scaled_features_test)

count = 0

for i in range(len(l_test)):

    if result[i] == l_test[i]:
        count += 1

print("accuracy: ", (count/len(l_test))*100)

#print(f_train)
#print(scaled_features_train)

