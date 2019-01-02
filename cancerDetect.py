import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

start = time.time()



dataSet = pd.read_csv("BreastCancerData.txt")



f1n = np.array(dataSet["clumbThick"])
f1 = f1n.reshape(-1,1)

f2n = np.array(dataSet["unSize"])
f2 = f2n.reshape(-1,1)

f3n = np.array(dataSet["unShape"])
f3 = f3n.reshape(-1,1)

f4n = np.array(dataSet["mAdhesion"])
f4 = f4n.reshape(-1,1)

f5n = np.array(dataSet["SingleEpiSize"])
f5 = f5n.reshape(-1,1)

dataSet['BareNucli'] = dataSet['BareNucli'].replace('?', np.nan)
dataSet['BareNucli'] = dataSet['BareNucli'].fillna(0)
f6n = np.array(dataSet['BareNucli'])
f6n = f6n.astype(int)
f6 = f6n.reshape(-1,1)

f7n = np.array(dataSet["BalndChrome"])
f7 = f7n.reshape(-1,1)

f8n = np.array(dataSet["NormalNuclei"])
f8 = f8n.reshape(-1,1)

f9n = np.array(dataSet["mitosis"])
f9 = f9n.reshape(-1,1)

features = np.concatenate((f1, f2, f3, f4, f5, f6, f7, f8, f9), axis=1)

label = np.array(dataSet['label'])

features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2)

cancer_classifier = RandomForestClassifier(n_estimators=1000)

cancer_classifier.fit(features_train,label_train)

detect_result = cancer_classifier.predict(features_test)

count = 0

for i in range(len(detect_result)):

    if detect_result[i] == label_test[i]:

        count += 1

print('accuracy :' , count/len(detect_result)*100)

end = time.time()

print("features_train=\n", features_train)

print("label_train=\n", label_train)

print("features_test=\n", features_test)

print("label_test=\n", label_test)

print('run time= ', end-start)











