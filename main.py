import streamlit as st
import pickle
import pandas as pd


def load_model():
    with open('iris_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


lmodel = load_model()


def main():
    st.title("Iris Flower Prediction")
    st.markdown("\n####")
    sl = st.slider("Sepal Length", 4.0, 8.0)
    sw = st.slider("Sepal Width", 2.0, 5.0)
    pl = st.slider("Petal Length", 1.0, 7.0)
    pw = st.slider("Petal Width", 0.0, 3.0)

    if st.button('Predict'):
        pred = lmodel.predict([[sl, sw, pl, pw]])
        st.write(f"### The predicted species of the Flower is : {pred[0]}")


if __name__ == "__main__":
    main()