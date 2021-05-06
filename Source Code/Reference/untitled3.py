import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
#from datetime import datetime

Predict_Var = 3 #Dilihat dari cols (columns list)
df = pd.read_csv('dataMalangO.csv', nrows=365*30)
df = df.dropna()
#Separate dates for future plotting
# train_dates = pd.to_datetime(df['Date'])

#Variables for training
cols = list(df)[1:12]
encoder = LabelEncoder()
df_for_training = df[cols]
values = df_for_training.values
values[:,10] = encoder.fit_transform(values[:,10])
df_for_training = pd.DataFrame(values)
# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 2. We will make timesteps = 3. 
#With this, the resultant n_samples is 5 (as the input data has 9 rows).
trainX = []
trainY = []
testX = []
testY = []
ntraining = 365*5
train = len(df_for_training_scaled[:ntraining,-1])
test = len(df_for_training_scaled[ntraining:,-1])

n_future = 1   # Number of days we want to predict into the future
n_past = 1     # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    if i <= len(df_for_training_scaled[:ntraining,-1]):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, Predict_Var])
    else:
         testX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
         testY.append(df_for_training_scaled[i + n_future - 1:i + n_future, Predict_Var])   

trainX, trainY, testX, testY = np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))


# define Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit model
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1, shuffle=False)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, testX[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat1 = inv_yhat[:,0]
# invert scaling for actual
testY = testY.reshape((len(testY), 1))
inv_y = np.concatenate((testY, testX[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y1 = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y1, inv_yhat1))
print('Test RMSE: %.3f' % rmse)
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(df_for_training)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[n_past:ntraining+n_past, :] = inv_yhat
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(df_for_training)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[ntraining:, :] = inv_y
plt.plot(inv_y1, label='Training')
plt.plot(inv_yhat1, label='Test')
# plt.plot(inv_yhat1[0:1000])
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
plt.legend()
plt.show()

# #Forecasting...
# #Start with the last day in training date and predict future...
# n_future=90  #Redefining n_future to extend prediction dates beyond original n_future dates...
# forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

# forecast = model.predict(trainX[-n_future:]) #forecast 

# #Perform inverse transformation to rescale back to original range
# #Since we used 5 variables for transform, the inverse expects same dimensions
# #Therefore, let us copy our values 5 times and discard them after inverse transform
# forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
# y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]


# # Convert timestamp to date
# forecast_dates = []
# for time_i in forecast_period_dates:
#     forecast_dates.append(time_i.date())
    
# df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
# df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


# original = df[['Date', 'Open']]
# original['Date']=pd.to_datetime(original['Date'])
# original = original.loc[original['Date'] >= '2020-5-1']

# sns.lineplot(original['Date'], original['Open'])
# sns.lineplot(df_forecast['Date'], df_forecast['Open'])