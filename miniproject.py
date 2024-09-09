import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



# Title
st.title("Web App using Streamlit")

# Image
st.image("image.png", width=200)

st.title("Case Study on Diamond DataSets") 

# Load dataset
data = sns.load_dataset("diamonds")

st.write("Shape of the dataset:", data.shape)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Prediction Price"])

if menu == "Home":
    st.image("gold.png", width=200)
    st.header("Tabular Data of Diamonds")

    if st.checkbox("Tabular Data"):
        st.table(data.head(150))

    st.header("Statistic Summary of the DataFrame")

    if st.checkbox("Statistics"):
        st.table(data.describe())

    st.header("Correlation Graph")

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.title("Graphs")
    graph = st.selectbox("Different types of graphs", ["Scatter plot", "Bar Graph", "Histogram"])
    if graph == "Scatter plot":
        value = st.slider("Filter data using carat", 0, 6)
        data = data.loc[data["carat"] >= value]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=data, x="carat", y="price", hue="cut")
        st.pyplot(fig)

    if graph == "Bar Graph":
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="cut", y=data.cut.index, data=data)
        st.pyplot(fig)

    if graph == "Histogram":
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data.price, kde=True)
        st.pyplot(fig)

elif menu == "Prediction Price":
    st.title("Prediction price of a diamond")

    lr = LinearRegression()
    x = np.array(data["carat"]).reshape(-1, 1)
    y = np.array(data["price"]).reshape(-1, 1)
    lr.fit(x, y)

    # Input slider for carat value
    carat_value = st.slider("Select the carat value", min_value=0.0, max_value=5.0, step=0.1)
    predicted_price = lr.predict(np.array([[carat_value]]))
    st.write(f"The predicted price for a diamond with {carat_value} carat is ${predicted_price[0][0]:.2f}")