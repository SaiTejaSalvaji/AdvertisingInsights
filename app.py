# ---- Importing Libraries ----
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ---- Streamlit Config ----
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📊",
    layout="wide",
)

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("advertising.csv")
    return df

df = load_data()

# ---- Preprocessing ----
X = df[["TV", "radio", "newspaper"]]
Y = df["sales"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Load or Train Model ----
@st.cache_resource
def train_model():
    model = LinearRegression()
    model.fit(X_scaled, Y)
    joblib.dump(model, 'project.joblib')
    return model

model = train_model()

# ---- Title and Header ----
st.title("📈 Maximizing Returns")
st.subheader("Impact of Advertising Investments on Sales Revenue")

# ---- Sidebar Setup ----
image = Image.open("company.jpg")
st.sidebar.image(image, width=280)
st.sidebar.header("💡 Fusion Flicks")

# ---- Sidebar Inputs ----
st.sidebar.subheader("Enter Investment (in ₹1000):")

TV = st.sidebar.number_input("TV", min_value=0, step=10, value=0, format="%d")
radio = st.sidebar.number_input("Radio", min_value=0, step=10, value=0, format="%d")
newspaper = st.sidebar.number_input("Newspaper", min_value=0, step=10, value=0, format="%d")

# ---- Prediction Button ----
if st.sidebar.button("📊 Predict Sales"):
    if TV == 0 and radio == 0 and newspaper == 0:
        st.warning("⚠️ Please enter at least one non-zero investment value.")
    else:
        try:
            # Convert to DataFrame to match training data structure
            input_data = pd.DataFrame([[TV, radio, newspaper]], columns=["TV", "radio", "newspaper"])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            predicted_sales = round(prediction, 2)

            # ---- Display Prediction ----
            st.metric(label="📈 Predicted Sales (in ₹1000)", value=f"₹{predicted_sales:.2f}K")

            # ---- Dynamic Gauge Chart ----
            max_sales = df['sales'].max() + 5
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_sales,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Sales Revenue"},
                number={'prefix': "₹", 'suffix': "K"},
                gauge={
                    'axis': {'range': [0, max_sales]},
                    'bar': {'color': '#636EFA'},
                    'steps': [
                        {'range': [0, max_sales * 0.25], 'color': '#FF4B4B'},
                        {'range': [max_sales * 0.25, max_sales * 0.75], 'color': '#FECB52'},
                        {'range': [max_sales * 0.75, max_sales], 'color': '#00CC96'}
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 2},
                        'thickness': 0.75,
                        'value': predicted_sales
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # ---- Investment Breakdown (Bar Chart) ----
            st.subheader("📊 Investment Breakdown by Channel")
            spending = [TV, radio, newspaper]
            labels = ["TV", "Radio", "Newspaper"]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, spending, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel("Investment (in ₹1000)")
            ax.set_title("Advertising Investment by Channel")
            st.pyplot(fig)

            # ---- Percentage of Investment (Pie Chart) ----
            st.subheader("🍰 Percentage of Total Investment by Channel")
            if sum(spending) > 0:
                pie_data = pd.DataFrame({'Channel': labels, 'Amount': spending})
                fig = px.pie(
                    pie_data,
                    names='Channel',
                    values='Amount',
                    color='Channel',
                    color_discrete_map={'TV': '#1f77b4', 'Radio': '#ff7f0e', 'Newspaper': '#2ca02c'},
                    hole=0.4
                )
                fig.update_traces(textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            # ---- Scatter Plots ----
            st.subheader("📌 Relationship Between Ad Spending and Sales")

            fig = px.scatter(df, x="TV", y="sales", color="radio", size="newspaper")
            fig.update_layout(title="TV vs Sales", template="plotly_dark")
            st.plotly_chart(fig)

            fig = px.scatter(df, x="radio", y="sales", color="TV", size="newspaper")
            fig.update_layout(title="Radio vs Sales", template="plotly_dark")
            st.plotly_chart(fig)

            fig = px.scatter(df, x="newspaper", y="sales", color="TV", size="radio")
            fig.update_layout(title="Newspaper vs Sales", template="plotly_dark")
            st.plotly_chart(fig)

            # ---- 3D Scatter Plot ----
            st.subheader("🌐 Combined Effect of TV, Radio, and Newspaper")
            fig = px.scatter_3d(
                df, x="TV", y="radio", z="newspaper", color="sales",
                size="sales", size_max=15
            )
            st.plotly_chart(fig)

            # ---- Heatmap ----
            st.subheader("🔥 Correlation Heatmap")
            correlation = df.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # ---- Pair Plot ----
            st.subheader("📸 Pair Plot of All Variables")
            fig = sns.pairplot(df, diag_kind='kde', corner=True)
            st.pyplot(fig)

            # ---- Histograms ----
            st.subheader("📊 Distribution of Features")
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sns.histplot(df['TV'], kde=True, ax=ax[0], color='blue')
            sns.histplot(df['radio'], kde=True, ax=ax[1], color='orange')
            sns.histplot(df['newspaper'], kde=True, ax=ax[2], color='green')
            st.pyplot(fig)

            # ---- Box Plot ----
            st.subheader("📦 Box Plot of Investments")
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sns.boxplot(y=df['TV'], ax=ax[0], color='blue')
            sns.boxplot(y=df['radio'], ax=ax[1], color='orange')
            sns.boxplot(y=df['newspaper'], ax=ax[2], color='green')
            st.pyplot(fig)

            # ---- Residual Plot ----
            st.subheader("🔍 Residual Plot")
            predictions = model.predict(X_scaled)
            residuals = Y - predictions
            fig, ax = plt.subplots()
            sns.scatterplot(x=predictions, y=residuals, ax=ax)
            ax.axhline(0, color='red', linestyle='--')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ---- Display Raw Data ----
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

# ---- Footer ----
st.markdown("---")
st.markdown("👨‍💻 Built with ❤️ by [Fusion Flicks](https://www.instagram.com/fusion__flicks)")
