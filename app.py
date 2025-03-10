import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



df = pd.read_csv("advertising.csv")
# print(df.head())

X = df[["TV", "radio", "newspaper"]]
Y = df["sales"]

model = LinearRegression()
model.fit(X, Y)

# new_data = pd.DataFrame([[20, 324, 232]], columns=["TV", "radio", "newspaper"])
# prediction = model.predict(new_data)
# print("{:.2f}".format(prediction[0]))

st.set_page_config(
        page_title="Sales Prediciton",
        page_icon="logo.png",
        layout="wide",
    )

st.title("Maximizing Returns")
st.header("Impact of Advertising Investments on Sales Revenue")


image = Image.open("company.jpg")
st.sidebar.image(image)
st.sidebar.header("FUSION FLICKS")
st.sidebar.subheader("Enter investment in(in thousands of dollars):")
col1, col2, col3 = st.columns(3)
with col1:
    TV = st.sidebar.number_input("TV", min_value=0, step=10)

with col2:
    radio = st.sidebar.number_input("Radio", min_value=0, step=10)

with col3:
    newspaper = st.sidebar.number_input("Newspaper", min_value=0, step=10)


predict = st.sidebar.button("Predict Sales")

z = False
temp =0

if predict:
    if newspaper==0 and TV==0 and radio ==0:
        st.metric(label="Sales (in thousands of dollars)", value="0")
        z = True  

    if newspaper==0:
        X = df[["TV", "radio"]]
        model.fit(X, Y)
        prediction = model.predict([[TV, radio]])
    if TV==0:
        X = df[["radio", "newspaper"]]
        model.fit(X, Y)
        prediction = model.predict([[radio, newspaper]])
    if radio==0:
        X = df[["TV", "radio"]]
        model.fit(X, Y)
        prediction = model.predict([[TV, radio]])

    if radio==0 and TV == 0:
        X = df[["newspaper"]]
        model.fit(X, Y)
        prediction = model.predict([[newspaper]])

    if radio==0 and newspaper == 0:
        X = df[["TV"]]
        model.fit(X, Y)
        prediction = model.predict([[TV]])

    if TV==0 and newspaper == 0:
        X = df[["radio"]]
        model.fit(X, Y)
        prediction = model.predict([[radio]])

    else:
        X = df[["TV","radio","newspaper"]]
        model.fit(X, Y)
        prediction = model.predict([[TV,radio,newspaper]])


    if not z:
        temp = prediction[0]





import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = temp,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Predicted Sales Revenue"},
    number={'suffix': " K$"},
    gauge={
        'axis': {'range': [0, 40]},  
        'bar': {'color': 'white'}, 
        'steps': [
            {'range': [0, 10], 'color': 'red'},
            {'range': [10, 25], 'color': 'yellow'},
            {'range': [25, 40], 'color': 'green'}
        ],
        'threshold': {
            'line': {'color': 'black', 'width': 2},
            'thickness': 1,
            'value': temp  
        }
    }
    ))
st.plotly_chart(fig)




fig, ax = plt.subplots()
spending = [TV, radio, newspaper]
labels = ["TV", "Radio", "Newspaper"]
ax.bar(labels, spending, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel("Investment (in thousands of dollars)")
ax.set_title("Advertising Investment by Channel")

st.pyplot(fig)


joblib.dump(model, 'project.joblib')

