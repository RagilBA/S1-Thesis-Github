## Import Library ##
import pandas as pd

## Reading conditions on train and test data ##
TrainY = pd.read_csv('TrainConditions.csv')
TestY = pd.read_csv('TestConditions.csv')

## Reading parameter from regression for test in classification ##
TestTemperature = pd.read_csv('PredTemperature.csv')
TestHeatIndex = pd.read_csv('PredHeatIndex.csv')
TestPrecipitation = pd.read_csv('PredPrecipitation.csv')
TestWindSpeed = pd.read_csv('PredWindSpeed.csv')
TestWindDirection = pd.read_csv('PredWindDirection.csv')
TestVisibility = pd.read_csv('PredVisibility.csv')
TestCloudCover = pd.read_csv('PredCloudCover.csv')
TestRelativeHumidity = pd.read_csv('PredRelativeHumidity.csv')

## Append all parameter into one dataframe ##
TestParameter = TestTemperature
TestParameter['HeatIndex'] = TestHeatIndex['HeatIndex']
TestParameter['Precipitation'] = TestPrecipitation['Precipitation']
TestParameter['WindSpeed'] = TestWindSpeed['WindSpeed']
TestParameter['WindDirection'] = TestWindDirection['WindDirection']
TestParameter['Visibility'] = TestVisibility['Visibility']
TestParameter['CloudCover'] = TestCloudCover['CloudCover']
TestParameter['RelativeHumidity'] = TestRelativeHumidity['RelativeHumidity']
TestParameter.to_csv('TestParameter.csv', index = False)

