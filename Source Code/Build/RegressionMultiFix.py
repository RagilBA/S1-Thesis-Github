##Importing library##
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

##Inserting data as needed##
nskip = 12305+4336+4842
ndata = 4932
ntrain = 4497
nbatch = 30
df = pd.read_csv('dataMalang-Modified.csv', nrows = ndata, skiprows = nskip ) #, nrows=365*30, skiprow, skipfooter
df = df.dropna()
df.columns = ['Date', 'Temperature','HeatIndex','Precipitation','WindSpeed',
              'WindDirection','Visibility','CloudCover','RelativeHumidity',
              'Conditions']
####################################################################################
##Temperature##
## Reading data ##
Predict_Var = 0 #Choosing variable to do regression
Namefile = 'PredTemperature.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Temperature.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Temperature",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Heat Index##
## Reading data ##
Predict_Var = 1 #Choosing variable to do regression
Namefile = 'PredHeatIndex.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Heat Index.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Heat Index",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Precipitation##
## Reading data ##
Predict_Var = 2 #Choosing variable to do regression
Namefile = 'PredPrecipitation.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Precipitation.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Precipitation",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Wind Speed##
## Reading data ##
Predict_Var = 3 #Choosing variable to do regression
Namefile = 'PredWindSpeed.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Wind Speed.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Wind Speed",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Wind Direction##
## Reading data ##
Predict_Var = 4 #Choosing variable to do regression
Namefile = 'PredWindDirection.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Wind Direction.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Wind Direction",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Visibility##
## Reading data ##
Predict_Var = 5 #Choosing variable to do regression
Namefile = 'PredVisibility.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Visibility.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Visibility",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Cloud Cover##
## Reading data ##
Predict_Var = 6 #Choosing variable to do regression
Namefile = 'PredCloudCover.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Cloud Cover.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Cloud Cover",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)

####################################################################################
##Relative Humidity##
## Reading data ##
Predict_Var = 7 #Choosing variable to do regression
Namefile = 'PredRelativeHumidity.csv' 
# File Name: PredTemperature.csv, PredHeatIndex.csv, PredPrecipitation.csv, 
# PredWindSpeed.csv, PredWindDirection.csv, PredVisibility.csv, PredCloudCover.csv,
# PredRelativeHumidity.csv

## Separating dates ##
data_dates = pd.to_datetime(df['Date'])

## Choosing variabel to use ##
cols = list(df)[1:10]
encoder = OneHotEncoder()
df_for_training = df[cols]
values = df_for_training.values
valuesT = values[:,8]
valuesT = valuesT.reshape((valuesT.shape[0], 1))
valuesT = encoder.fit_transform(valuesT).toarray()
values = np.delete(values,8,1)
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
ntraining = ntrain
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
train_dates = pd.DataFrame(train_dates)
test_dates = pd.DataFrame(test_dates)
train_dates.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

## Create LSTM Model ##
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]),
               return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

## Fitting Model ##
history = model.fit(trainX, trainY, epochs=100, batch_size=nbatch, 
                    validation_data=(testX, testY), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

## Make prediction based on Model ##
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
yhat = yhat.reshape((yhat.shape[0]))
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((trainY.shape[0]))

## Revert parameter variable ##
inv_x = testX
inv_x = scaler.inverse_transform(inv_x)
inv_x = np.delete(inv_x, [8,9,10,11,12], axis = 1)

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
rmse = np.roots(mse)
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
plt.legend()
plt.savefig('Relative Humidity.png', dpi=250)
plt.show()
f = open('RMSE.txt','a')
print("Relative Humidity",mse, file=f)
f.close()

## Create training data for classification ##
inv_xtest = trainX
inv_xtest = scaler.inverse_transform(inv_xtest)
inv_xtest = np.delete(inv_xtest, [8,9,10,11,12], axis = 1)
train_var_name = cols
train_var_name.remove('Conditions')
train_var_name.append('Dates')
train_x = pd.DataFrame(inv_xtest)
PrintTrain_var = train_x
PrintTrain_var['Dates'] = train_dates
PrintTrain_var.columns = [train_var_name]
PrintTrain_var = PrintTrain_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTrain_var.to_csv('TrainParameter.csv', index = False)

## Create actual data for classification ##
test_var_name = cols
test_x = pd.DataFrame(inv_x)
PrintTest_var = test_x
PrintTest_var['Dates'] = test_dates
PrintTest_var.columns = [test_var_name]
PrintTest_var = PrintTest_var[['Dates', 'Temperature','HeatIndex','Precipitation',
                               'WindSpeed','WindDirection','Visibility',
                               'CloudCover','RelativeHumidity']]
PrintTest_var.to_csv('TestParameter.csv', index = False)

## Create predicted data for classification ##
pred_var_name = cols[Predict_Var]
pred_y = pd.DataFrame(inv_yhat1)
PrintPred_var = pred_y
PrintPred_var['Date'] = test_dates
PrintPred_var.columns =[pred_var_name, 'Date']
PrintPred_var = PrintPred_var[['Date',pred_var_name]]
PrintPred_var.to_csv(Namefile, index = False) 
#Note: Change csv name per variable predicted (Temperature, HeatIndex, etc)

## Create actual conditions for train classification ##
train_cond = df['Conditions']
train_cond = train_cond[:ntraining]
train_cond = pd.DataFrame(train_cond)
train_cond.reset_index(drop=True, inplace=True)
PrintTrain_cond = train_cond
PrintTrain_cond['Date'] = train_dates
PrintTrain_cond.columns =['Conditions', 'Date']
PrintTrain_cond = PrintTrain_cond[['Date','Conditions']]
PrintTrain_cond.to_csv('TrainConditions.csv', index = False)

## Create actual conditions for test classification ##
act_cond = df['Conditions']
act_cond = act_cond[(ntraining+1):]
act_cond = pd.DataFrame(act_cond)
act_cond.reset_index(drop=True, inplace=True)
PrintAct_cond = act_cond
PrintAct_cond['Date'] = test_dates
PrintAct_cond.columns =['Conditions', 'Date']
PrintAct_cond = PrintAct_cond[['Date','Conditions']]
PrintAct_cond.to_csv('TestConditions.csv', index = False)