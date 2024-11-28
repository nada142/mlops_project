#!/usr/bin/env python
# coding: utf-8

import warnings
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
import dagshub
from mpl_toolkits.mplot3d import Axes3D

# Set tracking URI for MLflow
mlflow.set_tracking_uri("https://dagshub.com/nada142/mlops_project.mlflow")

# Initialize DAGsHub repo
dagshub.init(repo_owner='nada142', repo_name='mlops_project', mlflow=True)

# Configure MLflow to log locally
mlflow.set_experiment("recipe_clustering_experiment")


import pandas as pd

def split_data(input_file, chunk_size):
    data = pd.read_csv(input_file)
    # Split data into chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk.to_csv(f"chunk_{i // chunk_size + 1}.csv", index=False)

# Call the function
split_data("src/data/recipe_20.csv", 10)  # 100 rows per chunk

# Function to load and preprocess data
def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    
    # Parsing directions
    if isinstance(data['directions'].iloc[0], str):  # Check if parsing is needed
        data['directions_parsed'] = data['directions'].apply(lambda x: ast.literal_eval(x))
    else:
        data['directions_parsed'] = data['directions']

    # Extract cooking time
    data['cooking_time'] = data['directions_parsed'].apply(
        lambda directions: sum(extract_time(step) for step in directions) if isinstance(directions, list) else 0
    )

    # Parse other columns
    data['ingredients_parsed'] = data['ingredients'].apply(lambda x: ast.literal_eval(x))
    data['directions_parsed'] = data['directions'].apply(lambda x: ast.literal_eval(x))
    data['NER_parsed'] = data['NER'].apply(lambda x: ast.literal_eval(x))

    # Numeric features
    data['num_ingredients'] = data['ingredients_parsed'].apply(len)
    data['num_steps'] = data['directions_parsed'].apply(len)

    data_processed = data[['num_ingredients', 'num_steps', 'cooking_time']]
    return data, data_processed

# Function to extract cooking time from text
def extract_time(text):
    time_patterns = [
        r"(\d+)\s*(minutes|min)",       # Matches "20 minutes" or "20 min"
        r"(\d+)\s*(heures|hours|hrs|h)" # Matches "1 heure", "2 hours", etc.
    ]
    
    total_minutes = 0
    
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            value = int(match[0])
            unit = match[1].lower()
            
            if "heure" in unit or "hour" in unit:
                total_minutes += value * 60
            else:
                total_minutes += value
    return total_minutes

# Function to normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Function to run KMeans clustering and log results
def run_kmeans(data_normalized, n_clusters):
    with mlflow.start_run():
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_normalized)
        
        # Log model
        mlflow.sklearn.log_model(kmeans, "kmeans_model")

        # Log metrics
        silhouette_avg = silhouette_score(data_normalized, clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)
        
        # Log Calinski-Harabasz Score
        ch_score = calinski_harabasz_score(data_normalized, clusters)
        mlflow.log_metric("calinski_harabasz_score", ch_score)

        # Log clusters
        data_normalized['Cluster'] = clusters
        
        # Elbow method plot
        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_normalized)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 10), inertia, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.savefig("elbow_method.png")
        mlflow.log_artifact("elbow_method.png")

        # Visualization of KMeans clusters
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_normalized['num_ingredients'], data_normalized['num_steps'], data_normalized['cooking_time'], c=clusters, cmap='viridis')
        ax.set_title('3D Visualization of KMeans Clusters')
        ax.set_xlabel('Number of Ingredients')
        ax.set_ylabel('Number of Steps')
        ax.set_zlabel('Cooking Time')
        plt.savefig("kmeans_clusters.png")
        mlflow.log_artifact("kmeans_clusters.png")

        # Log additional cluster info
        mlflow.log_metric("inertia", kmeans.inertia_)

        return kmeans, clusters

# Function to run GMM clustering and log results
def run_gmm(data_normalized, n_components):
    with mlflow.start_run():
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm_clusters = gmm.fit_predict(data_normalized)
        
        # Log model
        mlflow.sklearn.log_model(gmm, "gmm_model")

        # Log metrics
        silhouette_avg = silhouette_score(data_normalized, gmm_clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)

        # Log Calinski-Harabasz Score
        ch_score = calinski_harabasz_score(data_normalized, gmm_clusters)
        mlflow.log_metric("calinski_harabasz_score", ch_score)

        # Log log-likelihood
        log_likelihood = gmm.score(data_normalized)
        mlflow.log_metric("log_likelihood", log_likelihood)
        
        # Log clusters
        data_normalized['GMM_Cluster'] = gmm_clusters
        
        # Visualization of GMM clusters
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_normalized['num_ingredients'], data_normalized['num_steps'], data_normalized['cooking_time'], c=gmm_clusters, cmap='cool')
        ax.set_title('3D Visualization of GMM Clusters')
        ax.set_xlabel('Number of Ingredients')
        ax.set_ylabel('Number of Steps')
        ax.set_zlabel('Cooking Time')
        plt.savefig("gmm_clusters.png")
        mlflow.log_artifact("gmm_clusters.png")

        return gmm, gmm_clusters

# Main function to orchestrate the data flow
def main(chunk_path):
    # Load and preprocess the data
    data, data_processed = load_and_preprocess_data(chunk_path)
    data_normalized = normalize_data(data_processed)

    # Run KMeans clustering and log results
    kmeans, kmeans_clusters = run_kmeans(data_normalized, n_clusters=3)

    # Run GMM clustering and log results
    gmm, gmm_clusters = run_gmm(data_normalized, n_components=4)
    
    # Visualizations
    # Save visualizations to be logged in DVC
    kmeans_fig_path = "kmeans_clusters.png"
    gmm_fig_path = "gmm_clusters.png"
    save_visualization(kmeans_clusters, data_normalized, kmeans_fig_path)
    save_visualization(gmm_clusters, data_normalized, gmm_fig_path)

    # Save the output
    return data

def save_visualization(clusters, data_normalized, output_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_normalized['num_ingredients'], data_normalized['num_steps'], data_normalized['cooking_time'], c=clusters, cmap='viridis')
    ax.set_title(f'3D Visualization of Clusters')
    ax.set_xlabel('Number of Ingredients')
    ax.set_ylabel('Number of Steps')
    ax.set_zlabel('Cooking Time')
    plt.savefig(output_path)

# Run the pipeline
if __name__ == "__main__":
    data_path = 'src/data/recipe_20.csv'  # Update path if necessary
    result = main(data_path)
