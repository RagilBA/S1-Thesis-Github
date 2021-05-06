from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('IRIS.csv')
print (df.head())
print(df['species'].unique())
df.isnull().values.any()

df['species'] = df['species'].map({'Iris-setosa' :0, 'Iris-versicolor' :1, 'Iris-virginica' :2}).astype(int) #mapping numbers
print (df.head())

plt.close();
sns.set_style('whitegrid');
sns.pairplot(df, hue='species', height=3);
plt.show()

sns.set_style('whitegrid');
sns.FacetGrid(df, hue='species', height=5) \
.map(plt.scatter, 'sepal_length', 'sepal_width') \
.add_legend();
plt.show()

x_data = df.drop(['species'],axis=1)
y_data = df['species']
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(x_data)
data = pd.DataFrame(X_data_minmax,columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print (df.head())

X_train, X_test, y_train, y_test = train_test_split(data, y_data,test_size=0.2, random_state = 0)
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)
ypred=knn_clf.predict(X_test) #These are the predicted output values

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print('Confusion Matrix:')
print(result)
result1 = classification_report(y_test, ypred)
print('Classification Report:',)
print (result1)
result2 = accuracy_score(y_test,ypred)
print('Accuracy:',result2)