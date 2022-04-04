import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
from PIL import Image

st.write("""
# EUR/GBP PRICES PREDICTION
# *euro vs great british pound*
""")
with st.form(key = 'my_form'):
    username = st.text_input('identifiant')
    password = st.text_input('mot de passe')
    st.form_submit_button('Login')
 
# connect to MetaTrader 5
if not mt5.initialize(server="MetaQuotes-Demo", login=57154335, password=""):
    print("initialize() failed, error code =",mt5.last_error())
    mt5.shutdown()
# st.write((mt5.terminal_info()))
# st.write((mt5.version()))
eurgbp_rates = mt5.copy_rates_from_pos("EURGBP", mt5.TIMEFRAME_M10, 0, 21)

mt5.shutdown() 
st.write('Les données de prédiction sont chargées')
df = pd.DataFrame(eurgbp_rates)


#Prepare Data

df.drop(['time', 'open', 'high', 'low', 'tick_volume', 'spread',
       'real_volume'], axis=1, inplace = True)

#st.dataframe(df)
data = list(df['close'])
if st.checkbox('Voir les données'):
    st.subheader('Les 20 dernières valeurs')
    st.write(data)

# preparing independent and dependent features
def prepare_data(timeseries_data, n_features):
	X, y =[],[]
	for i in range(len(timeseries_data)):
		# find the end of this pattern
		end_ix = i + n_features
		# check if we are beyond the sequence
		if end_ix > len(timeseries_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

#call the model
n_steps, n_features = 20, 1
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.load_weights('univ_v1_model.h5')

# demonstrate prediction for next 10 candles
x_input = np.array(data)
temp_input=list(x_input)
lst_output=[]
i=0
n_steps=20
n_features=1
while(i<10):
    
    if(len(temp_input)>3):
        x_input=np.array(temp_input[1:])
        st.write("{} candle input {}".format(i,x_input))
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        st.write("{} candle output {}".format(i,yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        st.write(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i=i+1
    
# Output
st.write("LA VALEUR A LA FERMETURE POUR LES 10 BOUGIES SUIVANTES:")
st.write(lst_output)

# Visualizing The Output
if st.checkbox('Show trend'):
    st.subheader('Visualisation de la prediction')
    st.write('La possible tendance de la prediction sur EUR/GBP est :')
    #st.line_chart(lst_output)
    image = Image.open('img.png')
    st.image(image, caption='eurgbp trend')









#st.line_chart(lst_output)