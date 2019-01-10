# Machine-learning-Intro
............
Data set are collected from UCI respiratory.
*link for breast cancer data set: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
link for blood transfusion data set: https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/
link for parkinson data set: https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification#
*Missing values exist.
............
cancerDetect.py is a classifier for the BreastCancerData.txt data set.
transfusion.py is a classifier for the blood trasfusion data set(blood.txt)
park.py is a classifier for the parkinson data set (pd_speech_features.csv)
...........
Doctor Predictor:
Doctor Predictor is GUI to train data sets on sklearn classifiers. It trains models through 4 sklearn classifiers(Random Forest, Linear Discriminant Analysis model, Gaussian model and Extra Tree Classifier).
The prediction is made through the model which gives maximum accuracy.
doctorPredictor.py is the python program to run the GUI.
pd_speech_features.csv is the dataset for parkinson disease, parkinson_patient.csv is the sample data for prediction.
blood.txt is the dataset for blood transfusion, bloodPatient.txt is the sample data for prediction.
BreastCancerData.txt is the data for breast cancer patients, BreastCancerPatient.txt is the sample data for prediction
key.txt contains information to upload data to the firebase server.
