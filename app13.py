import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


# APP TITLE

st.title("Global Development Clustering Dashboard")
st.write("""
This app groups countries based on key development indicators like **GDP**, 
**Life Expectancy**, **Internet Usage**, **CO₂ Emissions**, etc.  
Upload a dataset or use the default one.
""")


#  LOAD DATA

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    st.info("Using sample dataset (World Development Measurements)")
    try:
        df = pd.read_csv("World_development_mesurement.csv")
    except FileNotFoundError:
        st.error("Sample CSV file not found. Please upload your dataset.")
        st.stop()

st.write("### Preview of Data")
st.dataframe(df.head())


# DATA CLEANING & SCALING

st.subheader("Data Cleaning & Scaling")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
if len(numeric_cols) == 0:
    st.error("No numeric columns found.")
    st.stop()

df_numeric = df[numeric_cols]
imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

if "Country" in df.columns:
    df_scaled["Country"] = df["Country"]

st.write("Scaled data preview:")
st.dataframe(df_scaled.head())


#K-MEANS CLUSTERING

st.subheader("K-Means Clustering")

n_clusters = st.slider("Select number of clusters", 2, 10, 3)
cluster_features = df_scaled.select_dtypes(include=["float64", "int64"])
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(cluster_features)
df_scaled["Cluster"] = kmeans.labels_

st.write("Cluster labels added:")
st.dataframe(df_scaled.head())


# VISUALIZATION

st.subheader("Cluster Visualization")
x_axis = st.selectbox("Select X-axis", options=cluster_features.columns)
y_axis = st.selectbox("Select Y-axis", options=cluster_features.columns)

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=df_scaled,
    x=x_axis,
    y=y_axis,
    hue="Cluster",
    palette="viridis",
    ax=ax
)
plt.title(f"Clusters based on {x_axis} and {y_axis}")
st.pyplot(fig)


# CLUSTER STATISTICS

st.subheader("Cluster Summary")
numeric_cols_for_summary = df_scaled.select_dtypes(include=["float64", "int64"]).columns.tolist()
# Ensure 'Cluster' is included
if "Cluster" not in numeric_cols_for_summary:
    numeric_cols_for_summary.append("Cluster")

cluster_summary = df_scaled[numeric_cols_for_summary].groupby("Cluster").mean().round(2)
st.dataframe(cluster_summary)


# SEARCH BY COUNTRY

if "Country" in df_scaled.columns:
    st.subheader("Find Cluster for a Country")
    country_list = df_scaled["Country"].dropna().unique()
    selected_country = st.selectbox("Select a country", options=country_list)
    cluster_number = df_scaled.loc[df_scaled["Country"] == selected_country, "Cluster"].values[0]
    st.write(f"**{selected_country}** belongs to **Cluster {cluster_number}**")


#SAVE MODEL

st.subheader("Save Your Model")
if st.button("Save Model"):
    with open("kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    st.success("Model saved as kmeans_model.pkl")

st.success("App Ready! Run it using: streamlit run app.py")
