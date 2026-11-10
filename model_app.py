import joblib
import streamlit as st

model = joblib.load("regression.joblib")

size = st.number_input("Taille (m²)", min_value=0.0, value=50.0)
rooms = st.number_input("Nombre de chambres", min_value=0, value=2, step=1)
garden = st.number_input("Jardin (0 = non, 1 = oui)", min_value=0, max_value=1, value=0, step=1)
import pandas as pd
cols = list(model.feature_names_in_)
pred = model.predict(pd.DataFrame([[size, rooms, garden]], columns=cols))
st.write(f"Prix prédit : {pred[0]:.0f}")