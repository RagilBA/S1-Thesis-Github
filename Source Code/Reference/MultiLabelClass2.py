from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

## Reading data ##
df = pd.read_csv('dataMalangOE.csv') #nrows=365
df = df.drop(['Date'], axis = 1)
df.dropna(inplace=True)
print (df.head())
print(df['Conditions'].unique())
df.isnull().values.any()

## Separate result given (Conditions, y value) and data (x value)
x_data = df.drop(['Conditions'],axis=1)
y_data = df['Conditions']

## Scalling data ##
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(x_data)
data = pd.DataFrame(X_data_minmax,columns=['MaximumTemperature','MinimumTemperature','Temperature','HeatIndex','Precipitation','WindSpeed','WindDirection','Visibility','CloudCover','RelativeHumidity'])
print (data.head())

## Splitting data test and train ##
X_train, X_test, y_train, y_test = train_test_split(data, y_data,test_size=0.5, random_state = 0)

## Creating model ##
knn_clf=KNeighborsClassifier()

## Fitting and training model ##
knn_clf.fit(X_train,y_train)
ypred=knn_clf.predict(X_test) #These are the predicted output values

## Create confusion matrix as error analysis ##
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print('Confusion Matrix:')
print(result)
result1 = classification_report(y_test, ypred)
print('Classification Report:',)
print (result1)
result2 = accuracy_score(y_test,ypred)
print('Accuracy:',result2)