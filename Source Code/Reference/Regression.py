#Import Library
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

# load dataset
dataframe = read_csv('dataMalang2.csv', usecols=[3])
dataframe = dataframe.dropna()
look_back = 365
# proccdata = proccesing_dataset(dataframe, look_back)
# testXP = proccdata[1]
# trainXP = proccdata[0]
# inv_dataset = proccdata[2]

# fungsi untuk merubah timeseries menjadi supervised (univariate)
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def proccesing_dataset (dataframe, look_back):
    dataset = dataframe.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.25)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50 , activation='relu', 
                   input_shape=(trainX.shape[1], trainX.shape[2]),
                   return_sequences=True))
    model.add(LSTM(25 , activation='relu', return_sequences=False))
    # model.add(LSTM(150 , activation='relu', return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=50, batch_size=72, verbose=1)
    history = model.fit(trainX, trainY, epochs=20, batch_size=32,
                        validation_data=(testX, testY), verbose=1)
    trainX = model.predict(trainX)
    testX = model.predict(testX)
    # invert predictions
    trainX = scaler.inverse_transform(trainX)
    trainY = scaler.inverse_transform([trainY])
    testX = scaler.inverse_transform(testX)
    testY = scaler.inverse_transform([testY])
    inv_dataset = scaler.inverse_transform(dataset)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainX[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testX[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    print (history.history.keys())
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # shift train predictions for plotting
    trainXPlot = numpy.empty_like(dataset)
    trainXPlot[:, :] = numpy.nan
    trainXPlot[look_back:len(trainX)+look_back, :] = trainX
    # shift test predictions for plotting
    testXPlot = numpy.empty_like(dataset)
    testXPlot[:, :] = numpy.nan
    testXPlot[len(trainX)+(look_back*2)+1:len(dataset)-1, :] = testX
    # plot baseline and predictions
    plt.plot(inv_dataset)
    plt.plot(trainXPlot)
    plt.plot(testXPlot)
    plt.show()
    return trainXPlot, testXPlot, inv_dataset

# load dataset
dataframe = read_csv('dataMalang2.csv', usecols=[4], nrows=365*30)
dataframe = dataframe.dropna()
look_back = 1
proccdata = proccesing_dataset(dataframe, look_back)
testXP = proccdata[1]
trainXP = proccdata[0]
inv_dataset = proccdata[2]
