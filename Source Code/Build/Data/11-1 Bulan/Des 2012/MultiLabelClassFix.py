from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

## Reading data ##
trainX = pd.read_csv('TrainParameter.csv') 
trainX = trainX.drop(['Dates'], axis = 1)
trainX.dropna(inplace=True)

testX = pd.read_csv('TestParameter.csv') 
testX = testX.drop(['Date'], axis = 1)
testX.dropna(inplace=True)

trainY = pd.read_csv('TrainConditions.csv')
trainY = trainY.drop(['Date'], axis = 1)
trainY.dropna(inplace=True)

testY = pd.read_csv('TestConditions.csv')
testY = testY.drop(['Date'], axis = 1)
testY.dropna(inplace=True)

## Encoder Y ##
encoderTrain = LabelEncoder()
encoderTest = LabelEncoder()
trainY = encoderTrain.fit_transform(trainY)
testY = encoderTest.fit_transform(testY)
trainY = pd.DataFrame(trainY)
testY = pd.DataFrame(testY)

## Scalling data ##
MinMaxScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
trainXScale = MinMaxScaler.fit_transform(trainX)
testXScale = MinMaxScaler.fit_transform(testX)

X_train = pd.DataFrame(trainXScale, columns = [
                                                'Temperature',
                                                'HeatIndex',
                                                # 'Precipitation',
                                                'WindSpeed',
                                                # 'WindDirection',
                                                'Visibility',
                                                # 'CloudCover',
                                              'RelativeHumidity'
                                             ])
X_test = pd.DataFrame(testXScale, columns = [
                                               'Temperature',
                                                'HeatIndex',
                                                # 'Precipitation',
                                                'WindSpeed',
                                                # 'WindDirection',
                                                'Visibility',
                                                # 'CloudCover',
                                              'RelativeHumidity'
                                             ])
Y_train = trainY
Y_test = testY

## Creating model ##
knn_clf=KNeighborsClassifier()

## Fitting and training model ##
knn_clf.fit(X_train,Y_train)
Ypred=knn_clf.predict(X_test) #These are the predicted output values

## Create confusion matrix as error analysis ##
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, Ypred)
print('Confusion Matrix:')
print(result)
result1 = classification_report(Y_test, Ypred)
print('Classification Report:',)
print (result1)
result2 = accuracy_score(Y_test,Ypred)
print('Accuracy:',result2)
f = open('accuracyClassification.txt','a')
print("Confusion matrix\n",result, file=f)
print("Classification Report\n",result1, file=f)
print("Accuracy\n",result2, file=f)
f.close()
Yact = encoderTrain.inverse_transform(testY)
Ypred = encoderTrain.inverse_transform(Ypred)