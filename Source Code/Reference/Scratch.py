import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

variabletrain = 2
def scalling_data (dff): #Cut Variable and Scale data

    dfvalues = dff.values
    # cols_pred = list(dff)[11]
    cols_train = list(dff)[1:4]
    # encoder = LabelEncoder()
    # dfvalues[:,11] = encoder.fit_transform(dfvalues[:,11])
    # df = pd.DataFrame(dfvalues) #untuk condition
    df_for_training = dff[cols_train].astype(float)
    # df_for_pred = dff[cols_pred]
    # df_for_predenc = encoder.fit_transform(df_for_pred)
    # df = pd.DataFrame(df_for_predenc)
    # df_for_training['Conditions'] = df
    # LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    # df_for_training_scaled = np.hstack((df_for_training_scaled, np.atleast_2d(df_for_predenc).T))# As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    # In this example, the n_features is 2. We will make timesteps = 3.
    # With this, the resultant n_samples is 5 (as the input data has 9 rows).
    # trainX = []
    # trainY = []
    return df_for_training, df_for_training_scaled, scaler

def transform_series_to_supervised(df_for_training_scaled, n_past, n_future,variabletrain):
    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(n_past, len(df_for_training_scaled) - n_future + 1): #+1 karena range di max value nggak di include, mulai dari n_past karena data akan terpotong, dan jumlah n_future akan mengurangi data karena nilai x seriesnya akan memanjang ke kanan
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future,variabletrain])
    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY

dff = pd.read_csv('dataMalang2.csv')
dff.fillna(0, inplace=True)
train_dates = pd.to_datetime(dff['Date'])# Separate dates for future plotting
df_for_training = scalling_data(dff)[0]
df_for_training_scaled = scalling_data(dff)[1]
n_future = 1  # Number of days we want to predict into the future
n_past = 1  # Number of past days we want to use to predict the future
trainX = transform_series_to_supervised(df_for_training_scaled, n_future, n_past,variabletrain)[0]
trainY = transform_series_to_supervised(df_for_training_scaled, n_future, n_past,variabletrain)[1]
# print('trainX shape == {}.'.format(trainX.shape))
# print('trainY shape == {}.'.format(trainY.shape))

# define Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
# model.add(LSTM(8, activation='relu', return_sequences=False))
# model.add(Dropout(0.3))
model.add(Dense(1))
opt = optimizers.Adam(learning_rate = 0.002)
model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
history = model.fit(trainX, trainY, epochs=100, batch_size=365*2, validation_split=0.3, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
# Forecasting...
# Start with the last day in training date and predict future...
n_future = 365  # Redefining n_future to extend prediction dates beyond original n_future dates...
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:])  # forecast

# Perform inverse transformation to rescale back to original range
# Since we used 5 variables for transform, the inverse expects same dimensions
# Therefore, let us copy our values 5 times and discard them after inverse transform
scaler = scalling_data(dff)[2]
# encoder = scalling_data(dff)[3]
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,variabletrain]
# y_pred_future_round = np.floor(y_pred_future).astype(int)
# y_pred_future_string = encoder.inverse_transform(y_pred_future_round[:,9])
# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Conditions': y_pred_future})
# df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

# original = df[['Date', 'Conditions']]
# original['Date'] = pd.to_datetime(original['Conditions'])
# # original = original.loc[original['Date'] >= '2020-5-1']

# sns.lineplot(original['Date'], original['Conditions'])
# sns.lineplot(df_forecast['Date'], df_forecast['Conditions'])