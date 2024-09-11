import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load the features from the CSV generated in the feature extraction step
def load_features(csv_file):
    return pd.read_csv(csv_file)

# Custom scorer for silhouette score
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    if len(set(labels)) > 1:  # Avoid computing silhouette score for a single cluster
        return silhouette_score(X, labels)
    else:
        return -1  # Return a low score for single cluster results

# DBSCAN Parameter Tuning (eps, min_samples)
def tune_dbscan_params(features, param_grid, feature_cols):
    X = features[feature_cols].values

    # Setup GridSearch for DBSCAN
    dbscan_model = DBSCAN()
    grid_search = GridSearchCV(dbscan_model, param_grid, scoring=silhouette_scorer, cv=3, n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best DBSCAN Parameters: {best_params}")
    print(f"Best DBSCAN Silhouette Score: {best_score}")

    return best_params

# KMeans Parameter Tuning (n_clusters)
def tune_kmeans_params(features, param_grid, feature_cols):
    X = features[feature_cols].values

    # Setup GridSearch for KMeans
    kmeans_model = KMeans()
    grid_search = GridSearchCV(kmeans_model, param_grid, scoring='silhouette_score', cv=3, n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best KMeans Parameters: {best_params}")
    print(f"Best KMeans Silhouette Score: {best_score}")

    return best_params

# Visualize Clusters after Parameter Tuning
def visualize_clusters(features, labels, feature_cols):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=features[feature_cols[0]], y=features[feature_cols[1]], hue=labels, palette='Set1', s=50)
    plt.title("Cluster Visualization")
    plt.show()

# Main Function to Perform Parameter Tuning for DBSCAN and KMeans
def main():
    csv_file = 'lidar_features.csv'  # Path to the CSV file with extracted features
    features = load_features(csv_file)
    
    # Feature columns to use in clustering (adjust based on which features you want to use)
    feature_cols = ['height', 'density', 'roughness', 'linearity', 'planarity', 'sphericity']
    
    ### DBSCAN Parameter Tuning ###
    dbscan_param_grid = {
        'eps': np.arange(1, 10, 0.5),   # Epsilon values to test
        'min_samples': np.arange(2, 10, 1)  # Minimum number of points in a cluster
    }
    best_dbscan_params = tune_dbscan_params(features, dbscan_param_grid, feature_cols)
    
    ### KMeans Parameter Tuning ###
    kmeans_param_grid = {
        'n_clusters': np.arange(2, 15, 1)  # Range of cluster numbers to test
    }
    best_kmeans_params = tune_kmeans_params(features, kmeans_param_grid, feature_cols)
    
    # After parameter tuning, we can visualize the clusters with the best models
    # Using best DBSCAN model
    dbscan_model = DBSCAN(eps=best_dbscan_params['eps'], min_samples=best_dbscan_params['min_samples'])
    dbscan_labels = dbscan_model.fit_predict(features[feature_cols].values)
    
    visualize_clusters(features, dbscan_labels, feature_cols)

    # Using best KMeans model
    kmeans_model = KMeans(n_clusters=best_kmeans_params['n_clusters'])
    kmeans_labels = kmeans_model.fit_predict(features[feature_cols].values)
    
    visualize_clusters(features, kmeans_labels, feature_cols)

# Run the main function to perform parameter tuning and clustering
if __name__ == "__main__":
    main()
