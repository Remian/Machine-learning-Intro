from tkinter import *
from firebase import firebase
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


class guiML:

    def __init__(self, master):

        frame = Frame(master)


        self.diagnosis = StringVar()
        self.diagnosis.set("no diagnosis")
        self.statusVar = StringVar()
        self.statusVar.set("not trained")
        self.algoName = StringVar()
        self.algoName.set("not available")
        self.checkButtonState = IntVar()

        self.title = Label(frame, text="DOCTOR PREDICTOR")
        self.trainingFile_label = Label(frame, text="Training Data File")
        self.status_title = Label(frame, text="Status")
        self.status_var = Label(frame, textvariable=self.statusVar)
        self.predictionFile_label = Label(frame, text="Patient Data")
        self.diagnosis_label = Label(frame, text="Diagnosis/class")
        self.result_label = Label(frame, textvariable=self.diagnosis)
        self.algoUsed_label = Label(frame, text="Algorithm used for prediction")
        self.usedAlgo_label = Label(frame, textvariable=self.algoName)
        self.trainingFile_entry = Entry(frame)
        self.predictionFile_entry = Entry(frame)
        self.runButton = Button(frame, text="Train", bg="black", fg="white", command=self.trainAndPredict)
        self.predictButton = Button(frame, text="predict", bg="black", fg="white", command=self.predict)
        self.checkButton = Checkbutton(frame, text="upload data to firebase server", variable=self.checkButtonState)
        self.quitButton = Button(frame, text="QUIT", command=frame.quit)

        frame.pack(fill=BOTH,expand=TRUE)
        self.title.grid(row=0, column=1)
        self.trainingFile_label.grid(row=1, column=0, sticky=W)
        self.trainingFile_entry.grid(row=1, column=2, sticky=E)


        self.checkButton.grid(columnspan=3)
        self.runButton.grid(row=4, column=2, sticky=E)
        self.status_title.grid(row=5, column=0, sticky=W)
        self.status_var.grid(row=5, column=2, sticky=W)
        self.predictionFile_label.grid(row=6, column=0, sticky=W)
        self.predictionFile_entry.grid(row=6, column=2, sticky=E)
        self.predictButton.grid(row=7, column=2, sticky=E)
        self.diagnosis_label.grid(row=8, column=0, sticky=W)
        self.result_label.grid(row=8, column=2, sticky=W)
        self.algoUsed_label.grid(row=9, column=0, sticky=W)
        self.usedAlgo_label.grid(row=9, column=2, sticky=W)
        self.quitButton.grid(row=10, column=2, sticky=E)

        self.parkinsonForestClassifier = RandomForestClassifier(n_estimators=1000)
        self.ParkinsonLinearClassifier = LinearDiscriminantAnalysis()
        self.ParkinsonGbClassifier = GaussianNB()
        self.ParkinsonExtraTreeClassifier = ExtraTreesClassifier(n_estimators=1000)

        self.scoreCount = dict()
        self.timeCount = dict()
        self.classifiers = {self.parkinsonForestClassifier: 'RandomForestClassifier', self.ParkinsonLinearClassifier: 'LinearDiscriminantAnalysis', self.ParkinsonGbClassifier: 'GaussianNB', self.ParkinsonExtraTreeClassifier: 'ExtraTreesClassifier'}

        self.FILE = 'C:/Users/AA/PycharmProjects/appliedML/'
        self.KEYS = 'C:/Users/AA/PycharmProjects/appliedML/keys.txt'
        self.APP_DATABASE_URL = 'https://ecg-1-e1a65.firebaseio.com/'



    def upload_file(self, file_location, directory):
        fp = open(file_location)
        fp_keys = open(self.KEYS, "a")
        dat_file = ""
        for d in fp.readlines():
            dat_file += d
        ref = firebase.FirebaseApplication(self.APP_DATABASE_URL, None)
        data = {'file_name': 'data.dat',
                'data': dat_file
                }
        result = ref.post("/" + directory + "/", data)
        fp_keys.write(data['file_name'] + " " + result['name'] + '\n')




    def trainAndPredict(self):

        chButtonLogic = self.checkButtonState.get()
        trainingFile = self.trainingFile_entry.get()
        self.FILE = self.FILE+trainingFile

        dataSet = pd.read_csv(trainingFile)

        header = list(dataSet)

        header.pop(0)

        feature_header = list()


        lastIndex = len(header) - 1

        for i in header:

            if i is header[lastIndex]:

                label = np.array(dataSet[i])

            else:

                dataSet[i] = dataSet[i].replace('?', np.nan)
                dataSet[i] = dataSet[i].fillna(0)
                vars()[i] = np.array(dataSet[i])
                vars()[i] = vars()[i].reshape(-1, 1)

                feature_header.append(i)



        features = np.array([])

        for i in feature_header:

            if i is feature_header[0]:

                features = vars()[i]

            else:

                features = np.concatenate((features, vars()[i]), axis=1)


        featuresTrain, featuresTest, labelTrain, labelTest = train_test_split(features, label, test_size=0.2)

        min_max_scaler_train = minmax_scale(featuresTrain)
        scaled_features_train = maxabs_scale(min_max_scaler_train)

        min_max_scaler_test = minmax_scale(featuresTest)
        scaled_features_test = maxabs_scale(min_max_scaler_test)

        start = time.time()
        self.parkinsonForestClassifier.fit(scaled_features_train, labelTrain)
        end = time.time()
        score = self.parkinsonForestClassifier.score(scaled_features_test, labelTest)
        timeVar = end - start
        self.timeCount.update({self.parkinsonForestClassifier:timeVar})
        self.scoreCount.update({self.parkinsonForestClassifier:score})



        start = time.time()
        self.ParkinsonLinearClassifier.fit(scaled_features_train, labelTrain)
        end = time.time()
        score = self.ParkinsonLinearClassifier.score(scaled_features_test, labelTest)
        timeVar = end - start
        self.timeCount.update({self.ParkinsonLinearClassifier: timeVar})
        self.scoreCount.update({self.ParkinsonLinearClassifier: score})



        start = time.time()
        self.ParkinsonGbClassifier.fit(featuresTrain, labelTrain)
        end = time.time()
        score = self.ParkinsonGbClassifier.score(featuresTest, labelTest)
        timeVar = end - start
        self.timeCount.update({self.ParkinsonGbClassifier: timeVar})
        self.scoreCount.update({self.ParkinsonGbClassifier: score})


        start = time.time()
        self.ParkinsonExtraTreeClassifier.fit(scaled_features_train, labelTrain)
        end = time.time()
        score = self.ParkinsonExtraTreeClassifier.score(scaled_features_test, labelTest)
        timeVar = end - start
        self.timeCount.update({self.ParkinsonExtraTreeClassifier: timeVar})
        self.scoreCount.update({self.ParkinsonExtraTreeClassifier: score})

        if(chButtonLogic == 1):
            self.upload_file(self.FILE, 'ML data')
            self.statusVar.set("Trained")

        else:
            self.statusVar.set("Trained")



    def predict(self):


        sourceFile = self.predictionFile_entry.get()

        dataSet = pd.read_csv(sourceFile)

        dataSetNumpy = np.array(dataSet)
        dataSetNumpy = np.delete(dataSetNumpy, 0)
        features = np.delete(dataSetNumpy, -1)
        features = features.reshape(1,-1)

        result = max(self.scoreCount, key=self.scoreCount.get).predict(features)
        classifierName = self.classifiers.get(max(self.scoreCount, key=self.scoreCount.get))

        result = str(result).lstrip('[').rstrip(']')
        self.diagnosis.set(result+" (as per data label)")
        self.algoName.set(classifierName)






root = Tk()
c = guiML(root)
root.mainloop()


