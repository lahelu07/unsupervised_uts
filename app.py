import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset CPI
data_file = 'API_FP.CPI.TOTL_DS2_en_csv_v2_284.csv'
data = pd.read_csv(data_file, skiprows=4)

# Judul aplikasi
st.title("Pengelompokan Negara Berdasarkan CPI")

# Pilih model clustering
st.write("Langkah 1: Pilih model clustering")
model_choice = st.radio(
    "Pilih model clustering:",
    options=["KMeans (kmeans_model.pkl)", "Hierarchical Clustering (hc_model.pkl)"]
)

# Load model sesuai pilihan
if model_choice == "KMeans (kmeans_model.pkl)":
    model_file = 'kmeans_model.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    model_type = "KMeans"
    n_clusters = model.n_clusters
elif model_choice == "Hierarchical Clustering (hc_model.pkl)":
    model_file = 'hc_model.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    model_type = "Hierarchical Clustering"
    n_clusters = model.n_clusters

st.success(f"Model {model_type} berhasil dimuat dengan {n_clusters} kluster.")

# Pilih tahun untuk analisis
st.write("Langkah 2: Pilih 5 tahun untuk analisis CPI")
selected_years = st.multiselect(
    "Pilih 5 tahun:",
    options=[str(y) for y in range(1960, 2024)],
    default=[str(y) for y in range(2015, 2020)]  # Default 5 tahun terakhir
)

# Validasi jumlah tahun yang dipilih
if len(selected_years) != 5:
    st.error("Harap pilih tepat 5 tahun untuk melanjutkan.")
else:
    # Filter data berdasarkan tahun yang dipilih
    if all(y in data.columns for y in selected_years):
        filtered_data = data[["Country Name"] + selected_years].dropna()
        filtered_data = filtered_data.rename(columns={"Country Name": "Negara"})

        # Ambil nilai CPI sebagai data clustering
        cpi_values = filtered_data[selected_years].values

        # Lakukan clustering sesuai model yang dipilih
        if model_type == "KMeans":
            clusters = model.predict(cpi_values)
        elif model_type == "Hierarchical Clustering":
            clusters = model.fit_predict(cpi_values)

        # Validasi jumlah kluster sebelum menghitung Silhouette Score
        if len(set(clusters)) > 1:
            # Hitung Silhouette Score
            silhouette_avg = silhouette_score(cpi_values, clusters)
            st.write(f"Algoritma yang digunakan: {model_type}")
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        else:
            st.error("Tidak cukup kluster untuk menghitung Silhouette Score.")

        # Tambahkan kolom kluster ke data
        filtered_data["Cluster"] = clusters

        # Tampilkan data hasil klusterisasi
        st.write(f"Hasil klusterisasi negara berdasarkan CPI untuk tahun: {', '.join(selected_years)}")
        st.dataframe(filtered_data)

        # Visualisasi Kluster untuk setiap tahun
        st.write("Visualisasi Kluster untuk Setiap Tahun")
        for year in selected_years:
            fig, ax = plt.subplots()
            for cluster in set(clusters):
                cluster_data = filtered_data[filtered_data["Cluster"] == cluster]
                ax.scatter(
                    cluster_data["Negara"],
                    cluster_data[year],
                    label=f"Cluster {cluster}",
                    alpha=0.6
                )

            ax.set_title(f"Visualisasi Kluster - Tahun {year}")
            ax.set_xlabel("Negara")
            ax.set_ylabel("CPI")
            ax.legend()
            plt.xticks(rotation=90)  # Untuk memutar label negara
            st.pyplot(fig)

        # Opsional: Tampilkan statistik per kluster
        cluster_summary = filtered_data.groupby("Cluster")[selected_years].agg(["mean", "min", "max", "count"])
        st.write("Statistik per kluster:")
        st.dataframe(cluster_summary)
    else:
        st.error("Beberapa tahun yang dipilih tidak tersedia dalam dataset.")
