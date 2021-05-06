##Importing library##
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

## Reading data ##
Predict_Var = 1 #Menentukan variabel parameter yang ingin di regresikan
df = pd.read_csv('dataMalangO.csv') #, nrows=365*30
df = df.dropna()

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:12]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,10]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,10,1)
values = np.append(values, valuesT, axis = 1)
df_for_training = pd.DataFrame(values)

## Normalize the dataset with range 0-1 ##
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

##Splitting data to test and training##
trainX = []
trainY = []
testX = []
testY = []
ntraining = 365*24*3
train = len(df_for_training_scaled[:ntraining,-1])
test = len(df_for_training_scaled[ntraining:,-1])

n_future = 1   # Number of days we want to predict into the future
n_past = 1     # Number of past days we want to use to predict the future

##Create time series data ##
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    if i <= len(df_for_training_scaled[:ntraining,-1]):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, Predict_Var])
    else:
         testX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
         testY.append(df_for_training_scaled[i + n_future - 1:i + n_future, Predict_Var])   
trainX, trainY, testX, testY = np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

## Check  data shape ##
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

## Train Dates adjustment ##
train_dates = data_dates[:ntraining]
test_dates = data_dates[(ntraining+1):]
test_dates = pd.DataFrame(test_dates)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=180, validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))

## Revert scalling prediction ##
inv_yhat = testX
inv_yhat[:,Predict_Var] = yhat
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat1 = inv_yhat[:,Predict_Var] #Just taking predicted variable

## Revert scalling actual data ##
testY = testY.reshape((len(testY)))
inv_y = testX
inv_y[:,Predict_Var] = testY
inv_y = scaler.inverse_transform(inv_y)
inv_y1 = inv_y[:,Predict_Var] #Just taking predicted variable

## y1 = actual data result, yhat1 = predicted data result

## calculate MSE ##
mse = mean_squared_error(inv_y1, inv_yhat1)
print('Test MSE: %.3f' % mse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.show()

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv('pred_var_name.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('PrintAct_cond.csv', index = False)