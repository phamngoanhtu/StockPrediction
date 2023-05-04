import yfinance as yf
import pandas_datareader.data as pdr
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#LSTM
plt.style.use('bmh') #Bayesian Methods for Hackers style
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

st.title('This is the demo of the project "Stock Prediction"')


option = st.selectbox(
    'Please choose the stock you want to predict',
    ('AMD', 'CS', 'F', 'IMGN', 'NKE', 'TSLA', 'NVDA', 'UBER', 'INTC', 'META'))

st.write('You have selected:', str(option))

#Load data
yf.pdr_override()

prediction_days = 60

start_date = '01-01-2022'
end_date = '31-12-2022'

company = str(option)

start = datetime.strptime(start_date, '%d-%m-%Y')
end = datetime.strptime(end_date, '%d-%m-%Y')
df = pdr.get_data_yahoo(company, start=start, end=end)
df.round(2)

#Create a new dataframe with only the 'Close'
data = df.filter(['Close'])

#Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit_transform(data['Close'].values.reshape(-1,1))

#Load the test data
yf.pdr_override()

start_date = '01-01-2023'

currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year
end_date = str(currentDay) + '-' + str(currentMonth) + '-' + str(currentYear)

start = datetime.strptime(start_date, '%d-%m-%Y')
end = datetime.strptime(end_date, '%d-%m-%Y')
test_data = pdr.get_data_yahoo(company, start=start, end=end)

actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'],test_data['Close']), axis = 0)
inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

#Visualize the test data
st.title(company + ' stock price from 01/01/2023 to ' + end_date)

plt.figure(figsize=(40,16))
plt.title(company)
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(test_data['Close'])
st.pyplot(plt)

#Prediction on test data
x_test = []
for x in range(prediction_days, len(inputs)):
  x_test.append(inputs[x-prediction_days:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.models.load_model('./models/stock_' + company + '.h5')
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


#Predict the next week
week = []
for i in range(7):
  real_data = [inputs[len(inputs) + 1 - prediction_days:len(inputs+1),0]]
  if i > 0:
      real_data.append(prediction)
      real_data.pop(0)

  real_data = np.array(real_data)
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

  prediction = model.predict(real_data)
  week.append(prediction)
  

for i in range(7):
  week[i] = scaler.inverse_transform(week[i])

#Visualize the data
aa=float((prediction-inputs[-1])*100)
week = np.array(week).reshape(-1,1)
week = week.flatten()
st.title(f"Prediction of the next week's price")

plt.figure(figsize=(40,16))
plt.title(company + ' - Prediction of the next week')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(week)
st.pyplot(plt)