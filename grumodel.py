import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def readFile(company):

    print(company)
    match company:
        case "IBM":
            dataset = pd.read_csv('ExcelFiles/IBM/IBM.csv', index_col = 'Date', parse_dates=['Date'])
            print(dataset.head())
        case _:
            print("boop")

def prepareData(dataset):
    featureTrain, labelTrain, featureTest, labelTest = load_data(dataset, 'Close', Enrol_window, True)