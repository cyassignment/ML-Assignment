import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from minisom import MiniSom
import numpy as np
from tabulate import tabulate
from sklearn.cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D

def main():
    st.title('Diaster Area Frequency')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        # Read data from uploaded CSV file
        dataset = pd.read_csv(uploaded_file)

        # Select relevant features for PCA
        incident_dataframe = dataset[['state_label_encoded_robust_scaled', 'incident_type_label_encoded_robust_scaled']]

        # Perform K-means clustering on PCA-transformed data
        kmeans = KMeans(n_clusters=10, random_state=42)
        incident_dataframe['K means Cluster'] = kmeans.fit_predict(incident_dataframe)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(incident_dataframe)

        # Plot the clusters in 2D
        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster_num in range(10):
            subset = pca_data[incident_dataframe['K means Cluster'] == cluster_num]
            ax.scatter(subset[:, 0], subset[:, 1], label=f"Cluster {cluster_num}", alpha=0.6)

        ax.set_title('K-means Clustering with 2 Clusters (PCA visualization)')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Display the plot using Streamlit
        st.pyplot(fig)

        # Perform GMM clustering
        optimal_n_components = 10  # You need to define optimal_n_components
        best_gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
        best_gmm.fit(pca_data)
        cluster_labels_gmm = best_gmm.predict(pca_data)

        # Plot the clusters in 2D
        fig_gmm, ax_gmm = plt.subplots(figsize=(10, 6))
        for cluster_num in range(optimal_n_components):
            subset_gmm = pca_data[cluster_labels_gmm == cluster_num]
            ax_gmm.scatter(subset_gmm[:, 0], subset_gmm[:, 1], label=f"Cluster {cluster_num}", alpha=0.6)

        ax_gmm.set_title('Gaussian Mixture Model Clustering using PCA')
        ax_gmm.set_xlabel('Principal Component 1')
        ax_gmm.set_ylabel('Principal Component 2')
        ax_gmm.legend()
        ax_gmm.grid(True)

        # Display the GMM plot using Streamlit
        st.pyplot(fig_gmm)

        # Perform SOM clustering
        features_for_pca = pca_data[:, :2]
        som = MiniSom(10, 10, 2, sigma=1.0, learning_rate=0.5)
        som.random_weights_init(features_for_pca)
        som.train_random(features_for_pca, 100)
        clustered_data = som.win_map(features_for_pca)

        # Initialize plot
        plt.figure(figsize=(10, 10))

        # Assign colors to each cluster
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(clustered_data))))

        # Plot the clustered data
        for cluster, data_points in clustered_data.items():
            color = next(colors)
            data_points = np.array(data_points)
            plt.scatter(data_points[:, 0], data_points[:, 1], color=color, label=f'Cluster {cluster}')  # Add cluster label

        # Add legend
        plt.title('Self-Organizing Map (SOM) Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)

        # Display the SOM plot using Streamlit
        st.pyplot(plt)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.1, min_samples=3)
        incident_dataframe['DBSCAN Cluster'] = dbscan.fit_predict(pca_data)

        # Plot the clusters
        plt.figure(figsize=(10, 6))
        for cluster_num in incident_dataframe['DBSCAN Cluster'].unique():
            if cluster_num == -1:
                subset = pca_data[incident_dataframe['DBSCAN Cluster'] == cluster_num]
                plt.scatter(subset[:, 0], subset[:, 1], label=f"Outliers", color='black', alpha=0.6)
            else:
                subset = pca_data[incident_dataframe['DBSCAN Cluster'] == cluster_num]
                plt.scatter(subset[:, 0], subset[:, 1], label=f"Cluster {cluster_num}", alpha=0.6)

        plt.title('DBSCAN Clustering with PCA Visualization')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Display the DBSCAN plot using Streamlit
        st.pyplot(plt)

        # Perform Mean Shift clustering
        meanshift = MeanShift(bandwidth=0.5)  # Adjust the bandwidth value as needed
        cluster_labels_ms = meanshift.fit_predict(pca_data)

        # Create synthetic third dimension
        synthetic_third_dim = np.random.rand(len(pca_data))

        # Plot the clusters in 3D
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        for cluster_num in range(len(set(cluster_labels_ms))):
            subset = pca_data[cluster_labels_ms == cluster_num]
            ax_3d.scatter(subset[:, 0], subset[:, 1], synthetic_third_dim[cluster_labels_ms == cluster_num], label=f"Cluster {cluster_num}", alpha=0.6)

        # Set labels and title
        ax_3d.set_xlabel('Principal Component 1')
        ax_3d.set_ylabel('Principal Component 2')
        ax_3d.set_zlabel('Synthetic Third Dimension')
        ax_3d.set_title('Mean Shift Clustering with PCA Visualization (3D)')

        # Add legend
        ax_3d.legend()

        # Display the 3D plot using Streamlit
        st.pyplot(fig_3d)

if __name__ == "__main__":
    main()
