import time

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import maxabs_scale

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB


dataSet = pd.read_csv("pd_speech_features.csv")

header= list(dataSet)
#print(len(header))
header.pop(0)

feature_header = list()


#print(len(header))

lastIndex = len(header)-1

for i in header:

    if i is header[lastIndex]:

        label = np.array(dataSet[i])

    else:

        vars()[i] = np.array(dataSet[i])
        vars()[i] = vars()[i].reshape(-1,1)

        feature_header.append(i)



#print(feature_header)
#print(label)



features = np.array([])

for i in feature_header:

    if i is feature_header[0]:

        features = vars()[i]

    else:

        features = np.concatenate((features,vars()[i]),axis=1)


#print(features)
#print(label)

#print(features.shape)
#print(label.shape)

featuresTrain, featuresTest, labelTrain, labelTest = train_test_split(features,label, test_size= 0.2)

min_max_scaler_train = minmax_scale(featuresTrain)
scaled_features_train = maxabs_scale(min_max_scaler_train)

min_max_scaler_test = minmax_scale(featuresTest)
scaled_features_test = maxabs_scale(min_max_scaler_test)



start = time.time()

parkinsonForestClassifier = RandomForestClassifier(n_estimators=1000)

parkinsonForestClassifier.fit(scaled_features_train,labelTrain)

end = time.time()

result = parkinsonForestClassifier.predict(scaled_features_test)

score = parkinsonForestClassifier.score(scaled_features_test,labelTest)

print('accuracy Forest Classifier :', score*100)

print('model training time Forest Classifier = ', end-start)

startL = time.time()

ParkinsonLinearClassifier = LinearDiscriminantAnalysis()
ParkinsonLinearClassifier.fit(scaled_features_train,labelTrain)

endL = time.time()

accuracy = ParkinsonLinearClassifier.score(scaled_features_test,labelTest)

print('Linear Classifier accuracy = ', accuracy*100)

print('model training time Linear Classifier = ', endL-startL)


startG = time.time()

ParkinsonGbClassifier = GaussianNB()
ParkinsonGbClassifier.fit(featuresTrain,labelTrain)

endG = time.time()

accuracy = ParkinsonGbClassifier.score(featuresTest,labelTest)

print('Gausian Classifier accuracy = ', accuracy*100)

print('model training time gausian Classifier = ', endG-startG)


startE = time.time()

ParkinsonExtraTreeClassifier = ExtraTreesClassifier(n_estimators=1000)
ParkinsonExtraTreeClassifier.fit(scaled_features_train,labelTrain)

endE = time.time()

accuracy = ParkinsonExtraTreeClassifier.score(scaled_features_test,labelTest)

print('Extra Tree Classifier accuracy = ', accuracy*100)

print('model training time Extra Tree Classifier = ', endG-startG)









