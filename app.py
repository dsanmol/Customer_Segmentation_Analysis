import streamlit as st
import pandas as pd
import numpy as np
import joblib

model=joblib.load("kmeans_model.pkl")
scaler=joblib.load("scaler.pkl")
st.title("Customer Segmentation App")
st.write("Enter Details of customer")

Age=st.number_input("Age",min_value=18,max_value=100)
income=st.number_input("income",min_value=0,max_value=200000)
total_spend=st.number_input("total_spend",min_value=0,max_value=5000)
num_web_purchase=st.number_input("num_web_purchase",min_value=0,max_value=100)
num_store_purchase=st.number_input("num_store_purchase",min_value=0,max_value=100)
num_web_visit=st.number_input("num_web_visit",min_value=0,max_value=100)
recency=st.number_input("recency",min_value=0,max_value=100)

input_data=pd.DataFrame({
    "Age":[Age],
    "Income":[income],
    "total_spends":[total_spend],
    "NumWebPurchases":[num_web_purchase],
    "NumStorePurchases":[num_store_purchase],
    "NumWebVisitsMonth":[num_web_visit],
    "Recency":[recency]
})

input_scaled=scaler.transform(input_data)

if st.button("Predict Segment"):

    cluster=model.predict(input_data)[0]
    st.success(f"Customer Segment is Cluster {cluster}")
    st.write(
        """
        Cluster 0:High Recency,Mid Age Group
        
        Cluster 1:High Income,High Spending
        
        Cluster 2:Low Spends ,High Web visits
        
        Cluster 3:High Web and Store Purchase
        
        Cluster 4:High Web Purchase , Low Recency
        
        Cluster 5:High Store Purchase
        """
    )
