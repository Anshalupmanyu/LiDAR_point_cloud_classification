import laspy
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

# Load LiDAR Data from LAS file
def load_lidar_data(filename):
    las = laspy.read(filename)
    points = np.vstack((las.x, las.y, las.z)).transpose()  # Extract XYZ points
    intensity = las.intensity
    return points, intensity

# Compute Height Feature (Z-axis)
def compute_height(points):
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    heights = points[:, 2] - z_min  # Relative height
    return heights

# Compute Point Density (points per unit area)
def compute_density(points, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    densities = 1.0 / (np.mean(distances, axis=1) + 1e-8)  # Avoid division by zero
    return densities

# Compute Surface Roughness (local variance of heights)
def compute_surface_roughness(points, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    roughness = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        z_neighbors = points[indices[i], 2]
        roughness[i] = np.std(z_neighbors)  # Standard deviation of neighbors' heights
    return roughness

# Compute Convex Hull Volume (local area shape feature)
def compute_convex_hull(points, k=20):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    hull_volumes = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        try:
            hull = ConvexHull(points[indices[i]])
            hull_volumes[i] = hull.volume
        except Exception as e:
            hull_volumes[i] = 0  # If the convex hull fails, set volume to 0
    return hull_volumes

# Compute Eigenvalue Features (shape analysis using PCA)
def compute_eigenvalue_features(points, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    eigenvalues = np.zeros((points.shape[0], 3))
    
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        cov_matrix = np.cov(neighbors, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov_matrix)
        eigenvalues[i, :] = np.sort(eigvals)[::-1]  # Sort eigenvalues descending
    
    linearity = (eigenvalues[:, 0] - eigenvalues[:, 1]) / eigenvalues[:, 0]
    planarity = (eigenvalues[:, 1] - eigenvalues[:, 2]) / eigenvalues[:, 0]
    sphericity = eigenvalues[:, 2] / eigenvalues[:, 0]
    
    return linearity, planarity, sphericity

# Main Function to Extract Features
def extract_features(las_file):
    points, intensity = load_lidar_data(las_file)

    # Convert points to PyntCloud for some additional feature extraction
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    
    # Feature extraction
    height = compute_height(points)
    density = compute_density(points)
    roughness = compute_surface_roughness(points)
    hull_volume = compute_convex_hull(points)
    linearity, planarity, sphericity = compute_eigenvalue_features(points)

    # Create a DataFrame to store features
    feature_df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'height': height,
        'density': density,
        'roughness': roughness,
        'hull_volume': hull_volume,
        'linearity': linearity,
        'planarity': planarity,
        'sphericity': sphericity,
        'intensity': intensity
    })
    
    return feature_df

# Save features to a CSV file for later use
def save_features_to_csv(feature_df, output_csv):
    feature_df.to_csv(output_csv, index=False)

# Usage Example
las_file = 'pc.las'  # Path to your LiDAR file
output_csv = 'lidar_features.csv'

# Extract and save features
features = extract_features(las_file)
save_features_to_csv(features, output_csv)

print("Features extracted and saved to", output_csv)
