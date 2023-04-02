#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

#import data
data = pd.read_csv("Data Clean.csv")
image = Image.open("house3.jpg")
st.title("Forecasting Harga Rumah")
st.image(image, use_column_width=True)

#data check
st.write("Ini Adalah Sebuah Aplikasi Untuk Mengetahui kisaran Harga Rumah Yang Anda Pilih Menggunakan Machine Learning.")
check_data = st.checkbox("Lihat Simpel Data")
if check_data:
    st.write(data[1:6])
st.write("Sekarang mari kita temukan berapa harga ketika kita memilih beberapa parameter.")

#input data
sqft_liv = st.slider("Luas Ruang Tamu",int(data.sqft_living.min()),int(data.sqft_living.max()),int(data.sqft_living.mean()) )
sqft_abo = st.slider("Luas Ruang Tamu",int(data.sqft_above.min()),int(data.sqft_above.max()),int(data.sqft_above.mean()) )
bath     = st.slider("Berapa Banyak Kamar Mandi",int(data.bathrooms.min()),int(data.bathrooms.max()),int(data.bathrooms.mean()) )
view = st.slider("Pemandangan/Suasana",int(data.view.min()),int(data.view.max()),int(data.view.mean()) )
sqft_bas   = st.slider("Luas Ruang Bawah Tanah",int(data.sqft_basement.min()),int(data.sqft_basement.max()),int(data.sqft_basement.mean()) )
condition  = st.slider("Kondisi Rumah",int(data.condition.min()),int(data.condition.max()),int(data.condition.mean()) )

#splitting your data
X = data.drop('price', axis = 1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

#modelling step
#Linear Regression model
#import your model
model=LinearRegression()
#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[sqft_liv,sqft_abo,bath,view,sqft_bas,condition]])[0]
akurasi= np.sqrt(r2_score(y_test,model.predict(X_test)))

# Mendefinisikan nilai tukar USD ke IDR
usd_to_idr = 14000

# Mengkonversi harga rumah dari USD ke IDR
idr_predictions = predictions * usd_to_idr

# Menampilkan hasil konversi ke dalam tampilan Streamlit
if st.button("Mulai"):
    st.header("Prediksi Harga Rumah :")
    st.subheader("- USD {}".format(int(predictions)))
    st.subheader("- IDR {}".format(int(idr_predictions)))
    st.header("Prediksi Selisih :")
    st.subheader("-USD {} - USD {}".format(int(predictions-errors),int(predictions+errors)))
    st.subheader("-IDR {} - IDR {}".format(int((predictions-errors)*usd_to_idr),int((predictions+errors)*usd_to_idr)))
    st.header("Akurasi: {}".format(akurasi))
