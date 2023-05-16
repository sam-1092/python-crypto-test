import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf
import json
import flask
import time
import flask_cors

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

app = flask.Flask(__name__)
flask_cors.CORS(app)
@app.route('/', methods=['GET'])
def home_page() : 
    yf.pdr_override()
    query = str(flask.request.args.get('cur'))
    print(query)
    crypto_currency = query
    fiat_currency = 'USD'

    startdate = dt.datetime(2022,1,1)
    enddate = dt.datetime.now()

    data = pdr.get_data_yahoo(f'{crypto_currency}-{fiat_currency}', start=startdate, end=enddate)
    print(data)

    # Prepare data
    scaler =  MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    prediction_days = 30

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # Create the Neural network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    #Testing the Model
    test_start = dt.datetime(2022,1,1)
    test_end = dt.datetime.now()

    test_data = pdr.get_data_yahoo(f'{crypto_currency}-{fiat_currency}', start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'],test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)
    """
    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(prediction_prices, color='green', label='Predicted Prices')
    plt.title(f'{crypto_currency} Price Prediction')
    plt.xlabel("Time")
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()
    """
  
    # Predict next day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs)+1,0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    #print(actual_prices[0], type(actual_prices)) 
    #print(prediction_prices[0], type(actual_prices))
    json_actual = actual_prices.tolist()
    json_predicted = prediction_prices.tolist()
    final_j_p = []
    for i in json_predicted : 
        for j in i : 
            final_j_p.append(j)
    prediction = prediction.tolist()
    #print(type(json_actual), type(json_predicted))
    #print(prediction)
    final_json = {'actual' : json_actual, 'predicted' : final_j_p, 'next_day_prediction' : prediction[0]}
    return final_json
if __name__ == '__main__' : 
    app.run(port=7777)